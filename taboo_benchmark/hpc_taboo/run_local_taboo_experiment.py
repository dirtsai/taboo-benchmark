#!/usr/bin/env python3
"""
Taboo Game Benchmark Script - HuggingFace Version
Local model deployment version for HPC Slurm environments

Optimized for HuggingFace transformers models with quantization and parallelization support.

Dependencies:
- transformers: For model loading and inference
- torch: PyTorch backend
- bitsandbytes: For 4-bit/8-bit quantization (optional)

Features:
- Automatic 4-bit quantization to save GPU memory
- Chat template support for different model types
- Accurate token counting using model tokenizers
- HPC-optimized output paths (scratch directory)
- Parallel game execution with GPU-aware worker management
- Domain-aware prompting for specialized datasets
"""

import json
import random
import time
import logging
from datetime import datetime
import os
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    pipeline, set_seed, BitsAndBytesConfig
)
import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============ EXPERIMENT CONFIGURATION ============
EXPERIMENT_CONFIG = {
    "temperature": 0.7,      # Increased for more creativity
    "max_turns": 5,
    "hint_len": 20,          # Maximum hint length (token count)
    "domain": "general", 
    "max_reprompt_attempts": 3,
    "max_new_tokens": 100,   # Increased for longer responses
    "do_sample": True,
    "top_p": 0.9,           # Nucleus sampling: cumulative probability threshold
    "top_k": 50,            # Top-k sampling: consider top k most probable tokens
    "enable_quantization": True,  # Enable 4-bit quantization to save GPU memory
    "quantization_type": "4bit",  # Options: "4bit", "8bit", "none"
}

# Recommended open source models list (from most advanced to lightweight)
RECOMMENDED_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",     # Meta Llama 3.1
    "mistralai/Magistral-Small-2506",       
    "Qwen/Qwen3-8B",             
]

# Dataset mapping configuration
DATASET_MAPPING = {
    "general": "data/wordnet_dataset.json",
    "medical": "data/medical_wordnet_dataset.json", 
    "polysemous": "data/polysemous_wordnet_dataset.json",
    "biology": "data/biology_wordnet_dataset.json",
    "cs": "data/cs_wordnet_dataset.json",
    "law": "data/law_wordnet_dataset.json",
    "literature": "data/literature_wordnet_dataset.json"
}

# Prompt templates
HINTER_SYSTEM_PROMPT = """You are **Hinter** in a word-guessing Taboo Game in the **{domain}** domain.
– Your goal: help the Guesser identify the *target word* from the {domain} domain.
– Hard rules:
  1. **Do NOT** use or partially spell any taboo words.
  2. Produce **exactly one line** that starts with 'Clue: '.
  3. Keep the clue ≤{hint_len} tokens.
– After each turn you will get the Guesser's guess and must output the next clue.
– If you break a rule, the game is immediately lost.
– Consider domain-specific knowledge when providing clues for {domain} terms."""

HINTER_USER_TEMPLATE = """### Game context
Target word: **{target_word}**
Taboo list (must not appear in your clue): {taboo_words}

Previously given clues:
{history_clues}

Provide your next clue now."""

GUESSER_SYSTEM_PROMPT = """You are **Guesser** in a word-guessing Taboo Game.
– Read the latest clue and output **one single word** as your guess.
– Output must start with 'Guess: ' followed by your word.
– Respond with exactly one line; no explanations."""

GUESSER_USER_TEMPLATE = """### Game context
Latest clue from Hinter:
{latest_clue}

Your previous guesses (for context): {history_guesses}

Provide your next guess now."""


class HuggingFaceModelClient:
    """
    Hugging Face local model client supporting multiple open source large language models
    """
    
    def __init__(self, model_name: str, device: str = "auto", quantization_config=None):
        self.model_name = model_name
        self.device = device
        self.quantization_config = quantization_config
        
        logger.info(f"Loading HuggingFace model: {model_name}")
        
        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                padding_side="left"
            )
        except Exception as e:
            logger.error(f"Failed to load tokenizer for {model_name}: {e}")
            raise
        
        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        try:
            model_kwargs = {
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "device_map": "auto" if self.device == "cuda" else None,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True
            }
            
            # Add quantization config if provided
            if self.quantization_config is not None:
                model_kwargs["quantization_config"] = self.quantization_config
                logger.info(f"Using quantization: {self.quantization_config}")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        # Create pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        
        logger.info(f"Model loaded successfully, using device: {self.device}")
    
    def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate response based on messages
        
        Args:
            messages: List of conversation messages [{"role": "system", "content": "..."}, ...]
            **kwargs: Generation parameters
            
        Returns:
            Generated text response
        """
        # Build prompt
        prompt = self._build_prompt(messages)
        
        # Set generation parameters
        generation_kwargs = {
            "max_new_tokens": kwargs.get("max_new_tokens", EXPERIMENT_CONFIG["max_new_tokens"]),
            "temperature": kwargs.get("temperature", EXPERIMENT_CONFIG["temperature"]),
            "do_sample": kwargs.get("do_sample", EXPERIMENT_CONFIG["do_sample"]),
            "top_p": kwargs.get("top_p", EXPERIMENT_CONFIG["top_p"]),
            "top_k": kwargs.get("top_k", EXPERIMENT_CONFIG["top_k"]),
            "pad_token_id": self.tokenizer.eos_token_id,
            "return_full_text": False,
        }
        
        try:
            # Generate response
            outputs = self.generator(prompt, **generation_kwargs)
            response = outputs[0]["generated_text"].strip()
            
            # Clean output (remove possible special tokens)
            response = self._clean_response(response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return ""
    
    def _build_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Build prompt format suitable for the model using chat template"""
        try:
            # Try to use the model's chat template if available
            if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None:
                prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                return prompt
        except Exception as e:
            logger.warning(f"Failed to use chat template: {e}. Falling back to generic format.")
        
        # Fallback to generic format if chat template is not available
        prompt_parts = []
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                prompt_parts.append(f"System: {content}\n\n")
            elif role == "user":
                prompt_parts.append(f"User: {content}\n\n")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}\n\n")
        
        # Add assistant prompt to start generation
        prompt_parts.append("Assistant:")
        
        return "".join(prompt_parts)
    
    def _clean_response(self, response: str) -> str:
        """Clean model output"""
        # Remove possible role prefixes and extra whitespace
        response = response.replace("Assistant:", "").strip()
        response = response.replace("User:", "").strip()
        response = response.replace("System:", "").strip()
        
        # Keep only the first line (avoid model generating too much content)
        lines = response.split('\n')
        if lines:
            response = lines[0].strip()
        
        return response
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using HuggingFace tokenizer"""
        try:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            return len(tokens)
        except Exception:
            # Fallback to word count if tokenization fails
            return len(text.split())
    
    def get_tokenizer(self):
        """Get the HuggingFace tokenizer"""
        return self.tokenizer


def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Load dataset"""
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        logger.info(f"Successfully loaded dataset with {len(dataset)} entries")
        return dataset
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise


def validate_hint_format(hint_output: str) -> Tuple[bool, str]:
    """Validate hint format"""
    lines = hint_output.strip().split('\n')
    if not lines:
        return False, ""
    
    first_line = lines[0].strip()
    if not first_line.startswith("Clue: "):
        return False, ""
    
    hint = first_line[6:].strip()
    return True, hint


def validate_guess_format(guess_output: str) -> Tuple[bool, str]:
    """Validate guess format"""
    lines = guess_output.strip().split('\n')
    if not lines:
        return False, ""
    
    first_line = lines[0].strip()
    if not first_line.startswith("Guess: "):
        return False, ""
    
    guess = first_line[7:].strip()
    # Ensure it's a single word
    if len(guess.split()) != 1:
        return False, ""
    
    return True, guess


def check_taboo_violation(hint: str, taboo_words: List[str]) -> bool:
    """Check if taboo rules are violated"""
    hint_lower = hint.lower()
    
    for taboo in taboo_words:
        taboo_lower = taboo.lower()
        # Check exact word match
        if re.search(r'\b' + re.escape(taboo_lower) + r'\b', hint_lower):
            return True
        # Check partial spelling (at least 3 characters)
        if len(taboo_lower) >= 3 and taboo_lower in hint_lower:
            return True
    
    return False


def count_tokens_accurate(text: str, tokenizer) -> int:
    """Accurate token counting using actual tokenizer"""
    try:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)
    except Exception:
        # Fallback to word count if tokenization fails
        return len(text.split())

def count_tokens_rough(text: str) -> int:
    """Rough token counting (word-based) - kept for backward compatibility"""
    return len(text.split())


class TabooGame:
    """Taboo game engine"""
    
    def __init__(self, hinter_client: HuggingFaceModelClient, 
                 guesser_client: HuggingFaceModelClient, 
                 config: Dict[str, Any]):
        self.hinter_client = hinter_client
        self.guesser_client = guesser_client
        self.config = config
        
        # Game state
        self.turns = []
        self.history_clues = []
        self.history_guesses = []
        self.success = False
        self.total_tokens = 0
    
    def play_single_game(self, game_data: Dict[str, Any], dataset_name: str = "general") -> Dict[str, Any]:
        """Run a single game"""
        target_word = game_data['target']
        taboo_words = game_data['taboo']
        
        # Generate run ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{self.hinter_client.model_name.split('/')[-1]}_{self.guesser_client.model_name.split('/')[-1]}_{timestamp}_{random.randint(1000, 9999)}"
        
        logger.info(f"Starting game: {target_word} (taboo: {taboo_words})")
        
        # Reset game state
        self.turns = []
        self.history_clues = []
        self.history_guesses = []
        self.success = False
        self.total_tokens = 0
        
        # Run game turns
        for turn_id in range(1, self.config['max_turns'] + 1):
            logger.info(f"  Turn {turn_id}")
            
            turn_result = self._play_turn(turn_id, target_word, taboo_words, dataset_name)
            self.turns.append(turn_result)
            
            # Check for success or failure
            if not turn_result.get('hint_format_ok', True) or not turn_result.get('guesser_format_ok', True):
                logger.info("  Game failed due to format error")
                break
            
            if turn_result.get('correct', False):
                self.success = True
                logger.info(f"  Game success! Guessed correctly: {turn_result['guess']}")
                break
        
        # Calculate total tokens
        self.total_tokens = sum(
            turn.get('hint_tokens', 0) + turn.get('guess_tokens', 0) 
            for turn in self.turns
        )
        
        # Build result
        result = {
            "run_id": run_id,
            "hinter_model": self.hinter_client.model_name,
            "guesser_model": self.guesser_client.model_name,
            "temperature": self.config['temperature'],
            "domain": dataset_name,  # Use actual dataset name instead of fixed config value
            "target_word": target_word,
            "taboo_words": taboo_words,
            "success": self.success,
            "turn_count": len(self.turns),
            "total_tokens": self.total_tokens,
            "turns": self.turns
        }
        
        return result
    
    def _play_turn(self, turn_id: int, target_word: str, taboo_words: List[str], domain: str = "general") -> Dict[str, Any]:
        """Run a single turn"""
        # Get hint
        hint_result = self._get_hint(target_word, taboo_words, domain)
        
        if not hint_result['hint_format_ok']:
            return {
                "turn_id": turn_id,
                **hint_result,
                "guesser_prompt": "",
                "guesser_output": "",
                "guess": "",
                "guess_tokens": 0,
                "guesser_format_ok": False,
                "correct": False
            }
        
        # Get guess
        hint = hint_result['hint']
        self.history_clues.append(hint)
        
        guess_result = self._get_guess(hint)
        
        if not guess_result['guesser_format_ok']:
            return {
                "turn_id": turn_id,
                **hint_result,
                **guess_result,
                "correct": False
            }
        
        # Check if answer is correct
        guess = guess_result['guess']
        self.history_guesses.append(guess)
        correct = guess.lower() == target_word.lower()
        
        return {
            "turn_id": turn_id,
            **hint_result,
            **guess_result,
            "correct": correct
        }
    
    def _get_hint(self, target_word: str, taboo_words: List[str], domain: str = "general") -> Dict[str, Any]:
        """Get hint"""
        # Build history hints string
        history_clues_str = "\n".join([f"- {clue}" for clue in self.history_clues]) if self.history_clues else "None"
        
        # Build prompt
        system_prompt = HINTER_SYSTEM_PROMPT.format(hint_len=self.config['hint_len'], domain=domain)
        user_prompt = HINTER_USER_TEMPLATE.format(
            target_word=target_word,
            taboo_words=taboo_words,
            history_clues=history_clues_str
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Try multiple times (only for format errors)
        for attempt in range(self.config['max_reprompt_attempts']):
            try:
                # Call model
                hint_output = self.hinter_client.generate_response(
                    messages, temperature=self.config['temperature']
                )
                
                # Accurate token count using client's tokenizer
                hint_tokens = self.hinter_client.count_tokens(hint_output)
                
                # Validate format
                format_ok, hint = validate_hint_format(hint_output)
                if not format_ok:
                    logger.warning(f"Hint format error, attempt {attempt+1}: {hint_output}")
                    continue
                
                # Check length using accurate token count
                hint_token_count = self.hinter_client.count_tokens(hint)
                if hint_token_count > self.config['hint_len']:
                    logger.warning(f"Hint too long, attempt {attempt+1}: {hint_token_count} tokens")
                    continue
                
                # Check taboo violation (no retry)
                hint_violate = check_taboo_violation(hint, taboo_words)
                
                return {
                    "hinter_prompt": f"System: {system_prompt}\n\nUser: {user_prompt}",
                    "hinter_output": hint_output,
                    "hint": hint,
                    "hint_tokens": hint_tokens,
                    "hint_violate": hint_violate,
                    "hint_format_ok": True
                }
                
            except Exception as e:
                logger.error(f"Hint generation error, attempt {attempt+1}: {e}")
        
        # All attempts failed
        return {
            "hinter_prompt": f"System: {system_prompt}\n\nUser: {user_prompt}",
            "hinter_output": "",
            "hint": "",
            "hint_tokens": 0,
            "hint_violate": False,
            "hint_format_ok": False
        }
    
    def _get_guess(self, latest_hint: str) -> Dict[str, Any]:
        """Get guess"""
        # Build history guesses string
        history_guesses_str = ", ".join(self.history_guesses) if self.history_guesses else "None"
        
        # Build prompt
        user_prompt = GUESSER_USER_TEMPLATE.format(
            latest_clue=latest_hint,
            history_guesses=history_guesses_str
        )
        
        messages = [
            {"role": "system", "content": GUESSER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        # Try multiple times
        for attempt in range(self.config['max_reprompt_attempts']):
            try:
                # Call model
                guess_output = self.guesser_client.generate_response(
                    messages, temperature=self.config['temperature']
                )
                
                # Accurate token count using client's tokenizer
                guess_tokens = self.guesser_client.count_tokens(guess_output)
                
                # Validate format
                format_ok, guess = validate_guess_format(guess_output)
                if format_ok:
                    return {
                        "guesser_prompt": f"System: {GUESSER_SYSTEM_PROMPT}\n\nUser: {user_prompt}",
                        "guesser_output": guess_output,
                        "guess": guess,
                        "guess_tokens": guess_tokens,
                        "guesser_format_ok": True
                    }
                else:
                    logger.warning(f"Guess format error, attempt {attempt+1}: {guess_output}")
                    
            except Exception as e:
                logger.error(f"Guess generation error, attempt {attempt+1}: {e}")
        
        # All attempts failed
        return {
            "guesser_prompt": f"System: {GUESSER_SYSTEM_PROMPT}\n\nUser: {user_prompt}",
            "guesser_output": "",
            "guess": "",
            "guess_tokens": 0,
            "guesser_format_ok": False
        }


def validate_model_name(model_name: str) -> bool:
    """Validate if model name looks correct for HuggingFace"""
    # Basic validation for HuggingFace model names
    if "/" not in model_name:
        logger.warning(f"Model name {model_name} may not be a valid HuggingFace model")
        return False
    
    return True


def create_quantization_config(quantization_type: str):
    """Create quantization configuration"""
    if quantization_type == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif quantization_type == "8bit":
        return BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        return None


def run_single_game_wrapper(args_tuple):
    """Wrapper function for running a single game in parallel"""
    game_data, dataset_name, hinter_client, guesser_client, config = args_tuple
    
    try:
        # Create game instance
        game = TabooGame(hinter_client, guesser_client, config)
        
        # Run game
        result = game.play_single_game(game_data, dataset_name)
        result["dataset_name"] = dataset_name
        
        return result, None
    except Exception as e:
        logger.error(f"Game execution error: {e}")
        return None, str(e)


def run_experiment(hinter_models: List[str], guesser_models: List[str], 
                  datasets: List[str], output_dir: str, num_games_per_pair: int = 5,
                  max_workers: int = 4):
    """Run complete experiment"""
    
    # Validate model names
    for model in hinter_models + guesser_models:
        if not validate_model_name(model):
            logger.warning(f"Potential issue with model name: {model}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path(output_dir) / f"taboo_experiment_hf_{timestamp}"
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # Auto-adjust max_workers for GPU constraints
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        # For GPU inference, limit workers to avoid memory contention
        recommended_workers = min(max_workers, max(1, gpu_count * 2))
        if max_workers > recommended_workers:
            logger.warning(f"Reducing max_workers from {max_workers} to {recommended_workers} for GPU efficiency")
            max_workers = recommended_workers
    
    logger.info(f"Starting Hugging Face model experiment")
    logger.info(f"Output directory: {result_dir}")
    logger.info(f"Hinter models: {hinter_models}")
    logger.info(f"Guesser models: {guesser_models}")
    logger.info(f"Datasets: {datasets}")
    logger.info(f"Parallel workers: {max_workers} (GPU: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count() if torch.cuda.is_available() else 0})")
    
    # Load datasets
    dataset_dict = {}
    for dataset_name in datasets:
        if dataset_name in DATASET_MAPPING:
            dataset_path = DATASET_MAPPING[dataset_name]
            dataset_dict[dataset_name] = load_dataset(dataset_path)
            logger.info(f"Loaded dataset '{dataset_name}' from {dataset_path}: {len(dataset_dict[dataset_name])} entries")
        else:
            # Treat as file path
            dataset_dict[dataset_name] = load_dataset(dataset_name)
            logger.info(f"Loaded dataset from path '{dataset_name}': {len(dataset_dict[dataset_name])} entries")
    
    # Save experiment configuration
    experiment_params = {
        "experiment_name": "taboo_experiment_hf",
        "timestamp": timestamp,
        "hinter_models": hinter_models,
        "guesser_models": guesser_models,
        "datasets": datasets,
        "dataset_paths": {name: DATASET_MAPPING.get(name, name) for name in datasets},
        "num_games_per_pair": num_games_per_pair,
        **EXPERIMENT_CONFIG
    }
    
    config_file = result_dir / "experiment_config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(experiment_params, f, indent=2, ensure_ascii=False)
    
    # Store all results
    all_results = []
    conversation_data = []
    results_lock = Lock()  # For thread-safe access to shared data
    
    # Cache loaded models
    model_cache = {}
    model_cache_lock = Lock()  # For thread-safe model loading
    
    def get_model_client(model_name: str) -> HuggingFaceModelClient:
        with model_cache_lock:
            if model_name not in model_cache:
                # Create quantization config if enabled
                quantization_config = None
                if EXPERIMENT_CONFIG.get("enable_quantization", False):
                    quantization_config = create_quantization_config(
                        EXPERIMENT_CONFIG.get("quantization_type", "4bit")
                    )
                
                model_cache[model_name] = HuggingFaceModelClient(
                    model_name, 
                    quantization_config=quantization_config
                )
            return model_cache[model_name]
    
    total_combinations = len(hinter_models) * len(guesser_models) * len(datasets)
    combination_count = 0
    
    # Iterate through all model pairs and datasets
    for dataset_name, dataset in dataset_dict.items():
        logger.info(f"Processing dataset: {dataset_name} ({len(dataset)} entries)")
        
        for hinter_model in hinter_models:
            for guesser_model in guesser_models:
                combination_count += 1
                logger.info(f"Processing combination {combination_count}/{total_combinations}: {hinter_model} -> {guesser_model} on {dataset_name}")
                
                # Get model clients
                hinter_client = get_model_client(hinter_model)
                guesser_client = get_model_client(guesser_model)
                
                # Select game samples
                game_samples = random.sample(dataset, min(num_games_per_pair, len(dataset)))
                
                # Prepare arguments for parallel execution
                game_args = [
                    (game_data, dataset_name, hinter_client, guesser_client, EXPERIMENT_CONFIG)
                    for game_data in game_samples
                ]
                
                # Run games in parallel
                logger.info(f"  Running {len(game_samples)} games in parallel with {max_workers} workers")
                
                def process_result(result):
                    """Process a single game result"""
                    if result is None:
                        return
                    
                    with results_lock:
                        all_results.append(result)
                        
                        # Extract conversation data
                        for turn in result.get('turns', []):
                            # Hinter conversation
                            conversation_data.append({
                                'run_id': result['run_id'],
                                'hinter_model': result['hinter_model'],
                                'guesser_model': result['guesser_model'],
                                'target_word': result['target_word'],
                                'taboo_words': '|'.join(result['taboo_words']),
                                'turn_id': turn['turn_id'],
                                'role': 'hinter',
                                'model': result['hinter_model'],
                                'dataset_name': dataset_name,
                                'prompt': turn.get('hinter_prompt', ''),
                                'raw_output': turn.get('hinter_output', ''),
                                'processed_output': turn.get('hint', ''),
                                'tokens': turn.get('hint_tokens', 0),
                                'format_ok': turn.get('hint_format_ok', True),
                                'violate_taboo': turn.get('hint_violate', False),
                                'correct': False,
                                'success': result['success'],
                                'final_turn_count': result['turn_count']
                            })
                            
                            # Guesser conversation
                            conversation_data.append({
                                'run_id': result['run_id'],
                                'hinter_model': result['hinter_model'],
                                'guesser_model': result['guesser_model'],
                                'target_word': result['target_word'],
                                'taboo_words': '|'.join(result['taboo_words']),
                                'turn_id': turn['turn_id'],
                                'role': 'guesser',
                                'model': result['guesser_model'],
                                'dataset_name': dataset_name,
                                'prompt': turn.get('guesser_prompt', ''),
                                'raw_output': turn.get('guesser_output', ''),
                                'processed_output': turn.get('guess', ''),
                                'tokens': turn.get('guess_tokens', 0),
                                'format_ok': turn.get('guesser_format_ok', True),
                                'violate_taboo': False,
                                'correct': turn.get('correct', False),
                                'success': result['success'],
                                'final_turn_count': result['turn_count']
                            })
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all games for this model combination
                    future_to_game = {
                        executor.submit(run_single_game_wrapper, args): i 
                        for i, args in enumerate(game_args)
                    }
                    
                    # Process completed games
                    completed = 0
                    for future in as_completed(future_to_game):
                        completed += 1
                        game_idx = future_to_game[future]
                        
                        try:
                            result, error = future.result()
                            if result is not None:
                                logger.info(f"  Completed game {completed}/{len(game_samples)}: {result['target_word']}")
                                process_result(result)
                            elif error:
                                logger.error(f"  Game {game_idx+1} failed: {error}")
                        except Exception as e:
                            logger.error(f"  Game {game_idx+1} execution error: {e}")
    
    # Save results
    logger.info("Saving experiment results...")
    
    # Save game summary
    summary_df = pd.DataFrame(all_results)
    summary_file = result_dir / "taboo_experiment_game_summary.csv"
    summary_df.to_csv(summary_file, index=False, encoding='utf-8')
    
    # Save conversation data
    conversation_df = pd.DataFrame(conversation_data)
    conversation_file = result_dir / "taboo_experiment_conversations.csv"
    conversation_df.to_csv(conversation_file, index=False, encoding='utf-8')
    
    # Save detailed JSON results
    json_file = result_dir / "taboo_experiment_detailed_results.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # Output basic statistics
    total_games = len(all_results)
    successful_games = sum(1 for r in all_results if r['success'])
    success_rate = successful_games / total_games if total_games > 0 else 0
    total_tokens = sum(r['total_tokens'] for r in all_results)
    
    logger.info(f"Experiment completed!")
    logger.info(f"Total games: {total_games}")
    logger.info(f"Successful games: {successful_games}")
    logger.info(f"Success rate: {success_rate:.1%}")
    logger.info(f"Total token consumption: {total_tokens:,}")
    logger.info(f"Results saved to: {result_dir}")
    
    return result_dir


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Taboo Game Benchmark - Hugging Face Version")
    
    parser.add_argument("--datasets", nargs="+", 
                       default=["general"],
                       help="List of datasets to use. Options: general, medical, polysemous, biology, cs, law, literature, or file paths")
    parser.add_argument("--output", type=str, 
                       default=os.environ.get("SCRATCH", "/mnt/scratch") + f"/{os.environ.get('USER', 'user')}/taboo_results",
                       help="Output directory (defaults to scratch directory in HPC environment)")
    parser.add_argument("--hinter-models", nargs="+", 
                       default=["meta-llama/Llama-3.1-8B-Instruct"],
                       help="List of hinter models")
    parser.add_argument("--guesser-models", nargs="+",
                       default=["meta-llama/Llama-3.1-8B-Instruct"], 
                       help="List of guesser models")
    parser.add_argument("--num-games", type=int, default=5,
                       help="Number of games per model pair")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--use-recommended", action="store_true",
                       help="Use recommended model list")
    parser.add_argument("--use-target-datasets", action="store_true",
                       help="Use your target datasets: general, medical, polysemous")
    parser.add_argument("--disable-quantization", action="store_true",
                       help="Disable quantization (may require more GPU memory)")
    parser.add_argument("--quantization-type", type=str, default="4bit",
                       choices=["4bit", "8bit", "none"],
                       help="Quantization type (4bit, 8bit, or none)")
    parser.add_argument("--max-workers", type=int, default=2,
                       help="Maximum number of parallel workers for game execution (default: 2, recommended: 1-4 for GPU)")
    
    args = parser.parse_args()
    
    # Validate and warn about high parallelism
    if args.max_workers > 8:
        logger.warning(f"High parallel workers ({args.max_workers}) may cause GPU memory issues or performance degradation")
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    set_seed(args.seed)
    
    # Update quantization settings based on args
    if args.disable_quantization or args.quantization_type == "none":
        EXPERIMENT_CONFIG["enable_quantization"] = False
    else:
        EXPERIMENT_CONFIG["enable_quantization"] = True
        EXPERIMENT_CONFIG["quantization_type"] = args.quantization_type
    
    # Process model lists
    if args.use_recommended:
        hinter_models = RECOMMENDED_MODELS  # Use all recommended models
        guesser_models = RECOMMENDED_MODELS
        logger.info("Using recommended model list")
    else:
        hinter_models = args.hinter_models
        guesser_models = args.guesser_models
    
    # Process dataset lists
    if args.use_target_datasets:
        datasets = ["general", "medical", "polysemous"]
        logger.info("Using target datasets: general, medical, polysemous")
    else:
        datasets = args.datasets
    
    logger.info(f"Starting experiment, random seed: {args.seed}")
    
    # Run experiment
    result_dir = run_experiment(
        hinter_models=hinter_models,
        guesser_models=guesser_models,
        datasets=datasets,
        output_dir=args.output,
        num_games_per_pair=args.num_games,
        max_workers=args.max_workers
    )
    
    print(f"\n实验完成！结果保存至: {result_dir}")


if __name__ == "__main__":
    main()