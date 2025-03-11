import os
from hera.workflows import Workflow, script, Parameter, Steps, Volume, ExistingVolume, Resources, Artifact, \
    NoneArchiveStrategy
from hera.workflows.models import VolumeMount
from hera.shared import global_config
import urllib3

urllib3.disable_warnings()

# Configure access address and Token.
global_config.host = os.getenv("ARGO_HOST")
global_config.token = os.getenv("ARGO_TOKEN")
global_config.verify_ssl = False

volume_mount = VolumeMount(name="workdir", mount_path="/mnt/vol")

@script(
    image="acr-multiple-clusters-registry.cn-hangzhou.cr.aliyuncs.com/serverless-argo/deepseek-finetune:v4",
    inputs=[
        Parameter(name="dataset-path", default="/mnt/vol/datasets"),  # Input parameter: dataset save path
    ],
    volume_mounts=[volume_mount])
def download_dataset():
    from datasets import load_dataset
    import os

    save_path = "{{inputs.parameters.dataset-path}}"

    print("Downloading dataset...")
    if not os.path.exists(save_path):
        dataset = load_dataset(
            "SylvanL/Traditional-Chinese-Medicine-Dataset-SFT",
            split="train"
        )
        dataset.save_to_disk(save_path)
    print(f"Dataset saved to {save_path}")

@script(
    image="acr-multiple-clusters-registry.cn-hangzhou.cr.aliyuncs.com/serverless-argo/deepseek-finetune:v4",
    inputs=[
        Parameter(name="base-model", default="/mnt/vol/model"),  # Input parameter: dataset save path
        Parameter(name="model-name", default="model-name")
    ],
    volume_mounts=[volume_mount])
def download_model():
    from huggingface_hub import snapshot_download
    # Define download path
    download_path = "{{inputs.parameters.base-model}}"
    if not os.path.exists(download_path):
        snapshot_download(
            repo_id="unsloth/{{inputs.parameters.model-name}}",
            local_dir=download_path,
            ignore_patterns=["*.msgpack", "*.h5", "*.tflite"],  # Ignore unnecessary file formats
        )
    print(f"Model downloaded to {download_path}")

@script(
    image="acr-multiple-clusters-registry.cn-hangzhou.cr.aliyuncs.com/serverless-argo/deepseek-finetune:v4",
    inputs=[
        Parameter(name="dataset-path", default="/mnt/data/datasets"),  # Input parameter: dataset save path
        Parameter(name="base-model", default="/mnt/data"),
        Parameter(name="format-path", value="/mnt/data/format"),  # Optional parameter: message prompt
    ],
    resources=Resources(cpu_request=16, cpu_limit=16, memory_request="32Gi", memory_limit="32Gi"),
    volume_mounts=[volume_mount])
def format_prompts():
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import sys

    save_path = "{{inputs.parameters.base-model}}"
    format_path = "{{inputs.parameters.format-path}}"
    if os.path.exists(format_path):
        sys.exit(0)
    tokenizer = AutoTokenizer.from_pretrained(save_path)
    EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN

    dataset = load_from_disk("{{workflow.parameters.dataset-path}}")
    alpaca_prompt = """以下是描述任务的说明，并搭配提供更多上下文的输入。
    写出适当完成请求的回复。在回答之前，请仔细思考问题并创建循序渐进的思路链，以确保做出合乎逻辑且准确的回答。
    
    ### Instruction:
    您是一位在中医的临床推理、诊断和治疗计划等方面具有具有丰富经验的医学专家。请回答以下医学问题。
    
    ### Input:
    {}
    
    ### Response:
    {}"""

    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            text = alpaca_prompt.format( input, output) + EOS_TOKEN
            texts.append(text)
        return {"text": texts, }

    dataset = dataset.map(formatting_prompts_func, batched=True, )
    print("Formatting prompts")
    dataset.save_to_disk("{{inputs.parameters.format-path}}")

@script(
    image="acr-multiple-clusters-registry.cn-hangzhou.cr.aliyuncs.com/serverless-argo/deepseek-finetune:v4",
    inputs=[
        Parameter(name="format-path", default="/mnt/data/datasets"),  # Input parameter: dataset save path
        Parameter(name="model-path", value="Dataset download started"),  # Optional parameter: message prompt
        Parameter(name="output-path", value="")
    ],
    annotations={
        "k8s.aliyun.com/eci-use-specs": "ecs.gn7i-c16g1.4xlarge",
        "k8s.aliyun.com/eci-gpu-driver-version": "tesla=525.85.12",
    },
    resources=Resources(cpu_request=8, cpu_limit=12, memory_request="20Gi", memory_limit="40Gi", gpus=1),
    volume_mounts=[volume_mount])
def training():
    from unsloth import is_bfloat16_supported
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import load_from_disk
    import sys
    max_seq_length = 2048
    dataset = load_from_disk("{{inputs.parameters.format-path}}")
    base = "{{inputs.parameters.model-path}}"
    output_dir = "{{inputs.parameters.output-path}}"
    if os.path.exists(output_dir):
        sys.exit(0)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base,
        max_seq_length=4096,
        local_files_only=True,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=60,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
        ),
    )
    print("Fine-tuning model")
    trainer_stats = trainer.train()
    print(trainer_stats)

    save_path = "{{inputs.parameters.output-path}}"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

@script(
    image="acr-multiple-clusters-registry.cn-hangzhou.cr.aliyuncs.com/serverless-argo/deepseek-finetune:v4",
    inputs=[
        Parameter(name="model-path", default="/mnt/data/datasets"),  # 输入参数：数据集保存路径
    ],
    annotations={
        "k8s.aliyun.com/eci-use-specs": "ecs.gn7i-c16g1.4xlarge",
        "k8s.aliyun.com/eci-gpu-driver-version": "tesla=525.85.12",
    },
    resources=Resources(cpu_request=8, cpu_limit=12, memory_request="20Gi", memory_limit="40Gi", gpus=1),
    volume_mounts=[volume_mount])
def inference_template():
    from unsloth import FastLanguageModel
    modelpath = "{{inputs.parameters.model-path}}"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=modelpath,  # Local path
        max_seq_length=2048,  # Set the maximum length of the input sequence
        dtype=None,  # Data type (default auto-select, optional float16, bfloat16, etc.)
        load_in_4bit=True,  # Whether to load in 4-bit quantization (if enabled during saving)
    )
    prompt_style = """以下是描述任务的说明，并搭配提供更多上下文的输入。
    写出适当完成请求的回复。在回答之前，请仔细思考问题并创建循序渐进的思路链，以确保做出合乎逻辑且准确的回答。

    ### Instruction:
    您是一位在中医的临床推理、诊断和治疗计划等方面具有具有丰富经验的医学专家。请回答以下医学问题。

    ### Question:
    {}

    ### Response:
    {}"""
    question = "久咳不止怎么办？"
    FastLanguageModel.for_inference(model)
    inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda")
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=1200,
        use_cache=True,
    )
    response = tokenizer.batch_decode(outputs)
    outputs = response[0].split("### Response:")[1]
    if "think" in outputs:
        outputs = outputs.split("think>")[1]
    print(outputs)
    with open("/tmp/response.txt", "w", encoding="utf-8") as f:
        f.write(outputs)

with Workflow(
        name="tcm-deepseek-finetune-with-argo",
        entrypoint="main",
        namespace="default",
        arguments=[
            Parameter(name="dataset-path", value="/mnt/vol/traditional-chinese-medicine-data"),
            Parameter(name="format-path", value="/mnt/vol/traditional-chinese-medicine-fromat-data"),
            Parameter(name="base-model", value="/mnt/vol/deepseek-basemodel"),
            Parameter(name="output-model", value="/mnt/vol/deepseek-finetuned"),
            Parameter(name="model-name", value="DeepSeek-R1-Distill-Qwen-7B"),
        ],
        volumes=[ExistingVolume(name="workdir", claim_name="pvc-oss", mount_path="/mnt/vol")],
) as w:
    with Steps(name="main") as main:
        # Step 1: Download data/model
        with main.parallel():
            download_data_step = download_dataset(
                arguments={"dataset-path": "{{workflow.parameters.dataset-path}}"}
            )
            download_model_step = download_model(
                arguments={
                    "base-model": "{{workflow.parameters.base-model}}",
                    "model-name": "{{workflow.parameters.model-name}}",
                }
            )

        # Step 2: Format dataset
        format_prompts_step = format_prompts(
            arguments={
                "dataset-path": "{{workflow.parameters.dataset-path}}",
                "format-path": "{{workflow.parameters.format-path}}",
                "base-model": "{{workflow.parameters.base-model}}",
            }
        )

        # Step 3: Fine-tune model
        training_step = training(
            arguments={
                "format-path": "{{workflow.parameters.format-path}}",
                "model-path": "{{workflow.parameters.base-model}}",
                "output-path": "{{workflow.parameters.output-model}}"
            }
        )

        # Step 4: Parallel inference
        with main.parallel():
            inference_template(name="inference-finetuned", arguments={"model-path": "{{workflow.parameters.output-model}}"})
            inference_template(name="inference-basemodel", arguments={"model-path": "{{workflow.parameters.base-model}}"})

yaml_content = w.to_yaml()
# Save YAML to file
with open("tcm-finetune-argo-workflow.yaml", "w") as f:
    f.write(yaml_content)

w.create()