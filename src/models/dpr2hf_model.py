"""
This fast script transforms the output checkpoint file from DPR code to HuggingFace's
Based on DPR's own negerate_dense_embeddings and dense_retriever.py
It only works with camembert !
"""

import sys
import argparse
assert "dpr" in sys.modules, "You do not have the DPR module installed. Cannot do this conversion without it."

from dpr.utils.model_utils import setup_for_distributed_mode, get_model_obj, load_states_from_checkpoint,move_to_device

parser = argparse.ArgumentParser()
args = parser.parse_args()
DPR_CP_PATH = "path"

saved_state = load_states_from_checkpoint(DPR_CP_PATH)
set_encoder_params_from_state(saved_state.encoder_params, args)
print_args(args)

tensorizer, encoder, _ = init_biencoder_components(args.encoder_model_type, args, inference_only=True)

encoder_ctx = encoder.ctx_model
# encoder_question = encoder.



