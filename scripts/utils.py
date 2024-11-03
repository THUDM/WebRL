from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import DummyOptim, DummyScheduler, set_seed
from accelerate.utils import (
        DistributedDataParallelKwargs,
        DistributedType,
        GradientAccumulationPlugin,
        is_mlu_available,
        is_mps_available,
        is_npu_available,
        is_torch_version,
        is_xpu_available,
        load_fsdp_model,
        load_fsdp_optimizer,
        save_fsdp_model,
        save_fsdp_optimizer,
        DataLoaderConfiguration,
    )
from hparams import get_infer_args, get_train_args
import yaml
import functools
import inspect
from accelerate.utils import DeepSpeedPlugin
from transformers.integrations.deepspeed import HfTrainerDeepSpeedConfig


def propagate_args_to_deepspeed(accelerator, args, auto_find_batch_size=False):
        """
        Sets values in the deepspeed plugin based on the Trainer args
        """
        from transformers.integrations.deepspeed import HfTrainerDeepSpeedConfig

        ds_plugin = accelerator.state.deepspeed_plugin

        ds_plugin.hf_ds_config = HfTrainerDeepSpeedConfig(ds_plugin.hf_ds_config.config)
        ds_plugin.deepspeed_config = ds_plugin.hf_ds_config.config
        ds_plugin.hf_ds_config.trainer_config_process(args, auto_find_batch_size)
        

def create_accelerator_and_postprocess(training_args):
    grad_acc_kwargs = {}
    if training_args.accelerator_config.gradient_accumulation_kwargs is not None:
        grad_acc_kwargs = training_args.accelerator_config.gradient_accumulation_kwargs

    # check if num_steps is attempted to be passed in gradient_accumulation_kwargs
    if "num_steps" in grad_acc_kwargs and training_args.gradient_accumulation_steps > 1:
        # raise because we do not know which setting is intended.
        raise ValueError(
            "The `AcceleratorConfig`'s `num_steps` is set but `gradient_accumulation_steps` is greater than 1 in the passed `TrainingArguments`"
            "If using the passed `AcceleratorConfig` is desired, do not set the `TrainingArguments` `gradient_accumulation_steps`."
        )
    elif "num_steps" not in grad_acc_kwargs:
        # take the gradient_accumulation_steps setting from TrainingArguments.
        grad_acc_kwargs["num_steps"] = training_args.gradient_accumulation_steps

    grad_acc_kwargs["sync_with_dataloader"] = False

    gradient_accumulation_plugin = GradientAccumulationPlugin(**grad_acc_kwargs)

    accelerator_config = training_args.accelerator_config.to_dict()

    dataloader_config = DataLoaderConfiguration(
        split_batches=accelerator_config.pop("split_batches"),
        dispatch_batches=accelerator_config.pop("dispatch_batches"),
        even_batches=accelerator_config.pop("even_batches"),
        use_seedable_sampler=accelerator_config.pop("use_seedable_sampler"),
    )
    non_blocking = accelerator_config.pop("non_blocking")
    
    dataloader_config.non_blocking = non_blocking
    # this would have been updated above, no need for it anymore
    accelerator_config.pop("gradient_accumulation_kwargs")

    args = {
        "deepspeed_plugin": training_args.deepspeed_plugin,
        "gradient_accumulation_plugin": gradient_accumulation_plugin,
    }
    
    args["dataloader_config"] = dataloader_config
    
    # create accelerator object
    accelerator = Accelerator(**args)
    # some Trainer classes need to use `gather` instead of `gather_for_metrics`, thus we store a flag
    gather_function = accelerator.gather_for_metrics

    if "use_gather_object" in inspect.signature(gather_function).parameters.keys():
        gather_function = functools.partial(
            gather_function, use_gather_object=training_args.eval_use_gather_object
        )

    # deepspeed and accelerate flags covering both trainer args and accelerate launcher
    is_deepspeed_enabled = getattr(accelerator.state, "deepspeed_plugin", None) is not None
    is_fsdp_enabled = getattr(accelerator.state, "fsdp_plugin", None) is not None

    # post accelerator creation setup
    if is_fsdp_enabled:
        fsdp_plugin = accelerator.state.fsdp_plugin
        fsdp_plugin.limit_all_gathers = training_args.fsdp_config.get(
            "limit_all_gathers", fsdp_plugin.limit_all_gathers
        )
        
        fsdp_plugin.activation_checkpointing = training_args.fsdp_config.get(
            "activation_checkpointing", fsdp_plugin.activation_checkpointing
        )
        if fsdp_plugin.activation_checkpointing and training_args.gradient_checkpointing:
            raise ValueError(
                "The activation_checkpointing in FSDP config and the gradient_checkpointing in training arg "
                "can't be set to True simultaneously. Please use FSDP's activation_checkpointing logic "
                "when using FSDP."
            )

    if is_deepspeed_enabled and getattr(training_args, "hf_deepspeed_config", None) is None:
        propagate_args_to_deepspeed(accelerator, training_args)

    # `save_only_model` can't be used with DeepSpeed/FSDP along with `load_best_model_at_end`
    if (
        training_args.save_only_model
        and (is_deepspeed_enabled or is_fsdp_enabled)
        and training_args.load_best_model_at_end
    ):
        wrapper = "DeepSpeed" if is_deepspeed_enabled else "FSDP"
        raise ValueError(f"{wrapper} can't be used with `save_only_model` along with `load_best_model_at_end`.")

    # `auto_find_batch_size` isn't yet supported with DeepSpeed/FSDP
    if (is_deepspeed_enabled or is_fsdp_enabled) and training_args.auto_find_batch_size:
        wrapper = "DeepSpeed" if is_deepspeed_enabled else "FSDP"
        raise NotImplementedError(f"`{wrapper}` doesn't support `auto_find_batch_size`.")
    
    return accelerator

def get_accelerator(config_file='/workspace/qzh/LLaMA-Factory-policy/examples/train_full/llama3_full_policy_web.yaml'):
    args_path = config_file
    with open(args_path, 'r') as file:
        args = yaml.safe_load(file)
    # args 现在是一个 Python 字典
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)
    accelerator = create_accelerator_and_postprocess(training_args)
    
    training_args.hf_deepspeed_config = HfTrainerDeepSpeedConfig(training_args.deepspeed)
    training_args.hf_deepspeed_config.trainer_config_process(training_args)
    training_args.deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=training_args.hf_deepspeed_config)
    accelerator = create_accelerator_and_postprocess(training_args)
    
    propagate_args_to_deepspeed(accelerator, training_args, True)
    return accelerator

if __name__ == '__main__':
    get_accelerator()