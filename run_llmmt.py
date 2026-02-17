import logging
import os
import sys
import json
import numpy as np
import wandb

import datasets
import evaluate
import torch

import transformers
from transformers import (
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.utils import  send_example_telemetry
from utils.trainer_llmmt import LlmmtTrainer
from utils.utils import XML_TAGS, load_mmt_dataset, get_preprocessed_data, clean_outputstring, load_tokenizer, load_model, SavePeftModelCallback, get_key_suffix, NLLB_CODE
from utils.arguments import ModelArguments, DataTrainingArguments

logger = logging.getLogger(__name__)

def main():
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        # If we pass only one argument to the script and it's the path to a yaml file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_yaml_file(os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    wandb.init(
        project=data_args.wandb_project_name,
        name=data_args.wandb_run_name if data_args.wandb_run_name else os.path.basename(training_args.output_dir.rstrip("/")),
        config={
            **model_args.__dict__,
            **data_args.__dict__,
            **training_args.__dict__,
        },
    )

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_llmmt", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Training/evaluation parameters {training_args}")

    # Get the datasets
    pairs = data_args.language_pairs.split(",")
    train_raw_data, valid_raw_data, test_raw_data = {}, None, {}
    train_raw_data, valid_raw_data, test_raw_data = load_mmt_dataset(pairs, data_args, model_args, training_args, logger)
        
    # load tokenizer
    set_seed(training_args.seed)
    tokenizer = load_tokenizer(data_args, model_args, training_args, logger)
    
    if model_args.use_xml_tokens:
        num_added_toks = tokenizer.add_tokens(XML_TAGS)
        logger.info(f"We have added {num_added_toks} tokens.")
    
    shots_eval_dict = {}
    if data_args.few_shot_eval_path:
        for lg_pair in test_raw_data.keys():
            pair_shot_path = os.path.join(data_args.few_shot_eval_path, f"shots.{lg_pair}.json")
            if not os.path.isfile(pair_shot_path):
                ValueError(f"Make sure the language pair {lg_pair} is in the few shot eval folder!")
            with open(pair_shot_path) as f:
                shots_eval_dict[lg_pair] = json.load(f)

    if model_args.chat_style:
        dummy_sentence = "This is a dummy sentence"
        chat_dummy_sentence = [{"role": "user", "content": dummy_sentence}] 
        dummy_sentence_with_speical_tokens = tokenizer.apply_chat_template(chat_dummy_sentence, tokenize=False, add_generation_prompt=True)
        encoded = tokenizer.encode(dummy_sentence_with_speical_tokens, add_special_tokens=False)
        decoded_text = tokenizer.decode(encoded, skip_special_tokens=True)
        begin_prefix = decoded_text.split(dummy_sentence, 1)[0].strip()
        additional_suffix = decoded_text.split(dummy_sentence, 1)[-1]
    else:
        begin_prefix = ""
        additional_suffix = ""

    train_datasets, eval_datasets, test_datasets = get_preprocessed_data(train_raw_data, valid_raw_data, test_raw_data, pairs, tokenizer, shots_eval_dict, data_args, training_args, model_args)
    metric = evaluate.load("sacrebleu")

    # Load model
    model = load_model(data_args, model_args, training_args, tokenizer, logger)
    
    if model_args.use_xml_tokens:
        # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
        model.resize_token_embeddings(len(tokenizer))

    # if the model is encoder-decoder, force bos_token_id needs to be set
    if model_args.encoder_decoder_type == "nllb":
        # if all the pairs are to the same language
        if all([lg_pair.split("-")[1] == pairs[0].split("-")[1] for lg_pair in pairs]):
            model.config.forced_bos_token_id = tokenizer.convert_tokens_to_ids([NLLB_CODE[pairs[0].split("-")[1]]])
            logger.info(f"Set forced_bos_token_id to {model.config.forced_bos_token_id} corresponding to {pairs[0].split('-')[1]} for NLLB.")
        else:
            logger.warning("For NLLB, all pairs should be to the same language. Setting forced_bos_token_id to None.")
            model.config.forced_bos_token_id = None    
            
    # Initialize our Trainer
    trainer = LlmmtTrainer(
        model=model,
        args=training_args,
        train_dataset=train_datasets if training_args.do_train else None,
        eval_dataset=eval_datasets if training_args.do_eval else None,
        tokenizer=tokenizer,
        callbacks=[SavePeftModelCallback] if model_args.use_peft else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint

        trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_state()
        if model_args.use_peft and not model_args.unfrozen_layers:
            model.save_pretrained(training_args.output_dir) 
        else:
            trainer.save_model()  # Saves the tokenizer too for easy upload
    # Prediction
    if training_args.do_predict:
        trainer.args.prediction_loss_only = False
        
        lg_pairs = sorted(test_datasets.keys()) # make sure each device print in the same order
        for lg_pair in lg_pairs:
            test_dataset = test_datasets[lg_pair]
            src_lang, tgt_lang = lg_pair.split("-")
            logger.info(f"*** Prediction for {lg_pair}***")
            if model_args.encoder_decoder_type == "nllb":
                preds, _, _ = trainer.predict(
                test_dataset=test_dataset, 
                max_new_tokens=data_args.max_new_tokens, 
                num_beams=data_args.num_beams, 
                metric_key_prefix="test",
                use_cache=True,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids([NLLB_CODE[tgt_lang]]),
            )
            else:
                preds, _, _ = trainer.predict(
                    test_dataset=test_dataset, 
                    max_new_tokens=data_args.max_new_tokens, 
                    num_beams=data_args.num_beams, 
                    metric_key_prefix="test",
                    use_cache=True,
                )
            print(f"Predictions for {lg_pair} are {preds.shape}")
            # Replace -100s used for padding as we can't decode them
            if int(torch.cuda.current_device()) == 0:
                preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
                decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

                # Some simple post-processing
                decoded_preds = [pred.strip() for pred in decoded_preds]
                
                for idx in range(data_args.display_num_translations):
                    print("------------------------")
                    print(decoded_preds[idx])

                with open(os.path.join(training_args.output_dir, f"test-{src_lang}-{tgt_lang}{data_args.suffix_eval_file}"), "w", encoding="utf-8") as f:
                    suffix = get_key_suffix(tgt_lang, data_args, additional_suffix)
                    if len(shots_eval_dict) != 0:
                        split_idx = len(shots_eval_dict[lg_pair]) + 1
                    else:
                        split_idx = 1
                    for pred in decoded_preds:
                        # Output is itself if it is an encoder-decoder model, otherwise it is the prefix + output
                        pred = clean_outputstring(pred, suffix, logger, split_idx) if not model_args.encoder_decoder_type else pred.strip()
                        f.writelines([pred, "\n"])

if __name__ == "__main__":
    main()

