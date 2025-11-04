import argparse
import concurrent
from dotenv import load_dotenv
from tqdm import tqdm
import textgrad as tg
from textgrad.tasks import load_task
import numpy as np
import random
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
import json
import numpy as np
from tasks import load_task, parse_tag_answer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
load_dotenv(override=True)
from sklearn.metrics import accuracy_score, precision_score, f1_score

def evaluate_metrics(preds, labels, cm_labels, average='weighted'):
    preds = [parse_tag_answer(x) for x in preds]
    labels = [parse_tag_answer(x) for x in labels]
    
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average=average, zero_division=0)
    f1 = f1_score(labels, preds, average=average, zero_division=0)
    cm = confusion_matrix(labels, preds, labels=cm_labels)
    
    return {
        'accuracy': acc,
        'precision': precision,
        'f1': f1,
        'cm': cm
    }


def init(SYSTEM, CAUSAL_SYSTEM, llm_api_test, llm_api_eval, CAUSAL_SYSTEM_CONSTRAINT):
    system_prompt = tg.Variable(SYSTEM, 
                                requires_grad=True,
                                role_description="structured system prompt to a somewhat capable language model that specifies the behavior and strategies for the QA task.")
    causal_prompt = tg.Variable(CAUSAL_SYSTEM, 
                                requires_grad=True,
                                role_description="structured system prompt to a generate causal desscription for downstream model")

    model = tg.BlackboxLLM(llm_api_test, system_prompt)
    causal_model = tg.BlackboxLLM(llm_api_test, causal_prompt)
    optimizer = tg.TextualGradientDescent(engine=llm_api_eval, parameters=[system_prompt])
    optimizer_causal = tg.TextualGradientDescent(engine=llm_api_eval, parameters=[causal_prompt], constraints=[CAUSAL_SYSTEM_CONSTRAINT])
    return system_prompt, causal_prompt, model, causal_model, optimizer, optimizer_causal

def init_eval(val_set, test_set, eval_fn, model, causal_model, system_prompt, causal_prompt, cm_labels, iters=3):
    results = {"test_f1": [], "prompt": [], "validation_f1": [], 'system_prompt':[], 'causal_prompt': []}
    acc, t_rs, t_ys = eval_dataset_causal(test_set, eval_fn, model, causal_model, iters=iters)
    test_res = evaluate_metrics(t_rs, t_ys, cm_labels)
    results["test_f1"].append(test_res['f1'])

    acc, rs, ys = eval_dataset_causal(val_set, eval_fn, model, causal_model, iters=iters)
    val_res = evaluate_metrics(rs, ys, cm_labels)
    results["validation_f1"].append(val_res['f1'])
    results["system_prompt"].append(system_prompt.get_value())
    results["causal_prompt"].append(causal_prompt.get_value())
    print(f"SCG_val_f1: {val_res['f1']}, SCG_test_f1:{test_res['f1']}")
    return results, test_res, val_res

def evaluate_metrics(preds, labels, cm_labels, average='weighted'):
    preds = [parse_tag_answer(x) for x in preds]
    labels = [parse_tag_answer(x) for x in labels]
    
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average=average, zero_division=0)
    f1 = f1_score(labels, preds, average=average, zero_division=0)
    cm = confusion_matrix(labels, preds, labels=cm_labels)
    
    return {
        'accuracy': acc,
        'precision': precision,
        'f1': f1,
        'cm': cm
    }

def initialize_json_file(output_json):
    if not os.path.exists(output_json) or os.path.getsize(output_json) == 0:
        with open(output_json, 'w') as f:
            json.dump([], f, indent=4)
    else:
        with open(output_json, 'w') as f:
            json.dump([], f, indent=4)

def convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    else:
        return obj

def save_current_step(output_json, current_data):
    try:
        clean_data = convert_numpy(current_data)

        if os.path.exists(output_json):
            with open(output_json, 'r') as f:
                try:
                    existing_data = json.load(f)
                    if not isinstance(existing_data, list):
                        existing_data = [existing_data]
                except json.JSONDecodeError:
                    existing_data = []
        else:
            existing_data = []

        existing_data.append(clean_data)

        with open(output_json, 'w') as f:
            json.dump(existing_data, f, indent=4)

    except Exception as e:
        print(f"Error saving JSON: {e}")

def validate_and_update_prompt(optimizer, prompt_obj, prompt_key, val_set, eval_fn, model, causal_model, results, cm_labels, previous_performance=None, label="", iters=3):
    optimizer.step()
    acc, rs, ys = eval_dataset_causal(val_set, eval_fn, model, causal_model, iters=iters)
    val_res = evaluate_metrics(rs, ys, cm_labels)
    
    val_performance = np.mean(val_res['f1'])
    cur_val_performance = val_performance
    previous_performance = np.mean(results["validation_f1"][-1]) if previous_performance is None else previous_performance
    previous_prompt = results[prompt_key][-1]

    do_test = True

    if val_performance < previous_performance:
        prompt_obj.set_value(previous_prompt)
        val_performance = previous_performance
        print("Skip Test")
        do_test = False
        
  

    print(f"[{label} Validation] F1: {cur_val_performance:.4f}, Previous F1: {previous_performance:.4f}")
    print(f"[{label} Validation CM]:\n{np.array(val_res['cm'])}")
    return val_performance, do_test, val_res


def run_training(train_loader, val_set, test_set, eval_fn, model, causal_model, system_prompt, causal_prompt,
                 optimizer, optimizer_causal, results, cm_labels, output_json="results.json", **kwargs):
    total_epochs = kwargs.get("epoch", 1) 
    total_steps = kwargs.get("steps", 6) 
    iters = kwargs.get("iters", 3)
    test_f1 = None
    test_res = None
    for epoch in range(total_epochs):
        for steps, (batch_x, batch_y) in enumerate((pbar := tqdm(train_loader, position=0))):
            print(f"\nEpoch {epoch}, Step {steps}")
            optimizer.zero_grad()
            optimizer_causal.zero_grad()
            losses = []

            for (x, y) in zip(batch_x, batch_y):
                if isinstance(y, np.int64):
                    y = int(y)
                x = tg.Variable(x, requires_grad=False, role_description="query to the language model")
                y = tg.Variable(y, requires_grad=False, role_description="correct answer for the query")
                try:
                    causal_description = causal_model(x)
                    full_prompt = causal_description + x
                    response = model(full_prompt)
                except:
                    print("skip for None response")

                try:
                    eval_output_variable = eval_fn(inputs=dict(prediction=response, ground_truth_answer=y))
                except:
                    eval_output_variable = eval_fn([x, y, response])
                losses.append(eval_output_variable)

            total_loss = tg.sum(losses)
            total_loss.backward()

            # Validate and update system prompt
            val_performance, do_test_1, val_res_sys = validate_and_update_prompt(
                optimizer, system_prompt, "system_prompt", val_set, eval_fn, model, causal_model, results, cm_labels, label="System", iters=iters
            )

            # Validate and update causal prompt
            val_performance_causal, do_test_2, val_res_causal = validate_and_update_prompt(
                optimizer_causal, causal_prompt, "causal_prompt", val_set, eval_fn, 
                model, causal_model, results, cm_labels,previous_performance=val_performance, label="Causal", iters=iters
            )

            results["validation_f1"].append(val_performance_causal)

            if do_test_1 or do_test_2: 
                acc, t_rs, t_ys = eval_dataset_causal(test_set, eval_fn, model, causal_model, iters=iters)
                test_res = evaluate_metrics(t_rs, t_ys, cm_labels)
                test_f1 = test_res['f1']

                results["test_f1"].append(test_f1)
                results["system_prompt"].append(system_prompt.get_value())
                results["causal_prompt"].append(causal_prompt.get_value())

                print(f"[Test Result] F1: {np.mean(test_f1):.4f}")
            # print(f"[Test CM] F1:\n{np.array(test_res['cm'])}")
            else:
                if results["test_f1"] is not None:
                    test_f1 = results["test_f1"][-1]
                test_res = None
                results["test_f1"].append(test_f1)
                results["system_prompt"].append(system_prompt.get_value())
                results["causal_prompt"].append(causal_prompt.get_value())
                print("Skip Test")

            # Save only current step data (overwrite file)
            current_step_data = {
                "epoch": epoch,
                "step": steps,
                "validation_f1": val_performance_causal,
                "test_f1": test_f1,
                "system_prompt": system_prompt.get_value(),
                "causal_prompt": causal_prompt.get_value(),
                "val_result_system": val_res_sys,
                "val_result_causal": val_res_causal,
                "test_results": test_res
            }
            save_current_step(output_json, current_step_data)

            if steps == total_steps:
                break
    return results


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    
def eval_sample_causal(item, eval_fn, model, causal_model, causal_relations=None):
    """
    This function allows us to evaluate if an answer to a question in the prompt is a good answer.

    """
    x, y = item
    if isinstance(y, np.int64):
        y = int(y)
    x = tg.Variable(x, requires_grad=False, role_description="query to the language model")
    y = tg.Variable(y, requires_grad=False, role_description="correct answer for the query")
    if causal_relations:
        full_prompt = x + causal_relations
    else:
        causal_description = causal_model(x)
        full_prompt = causal_description + x
    response = model(full_prompt)
    try:
        eval_output_variable = eval_fn(inputs=dict(prediction=response, ground_truth_answer=y))
        return int(eval_output_variable.value), response.value, y.value
    except:
        eval_output_variable = eval_fn([x, y, response])
        eval_output_parsed = eval_fn.parse_output(eval_output_variable)
        return int(eval_output_parsed)

def eval_dataset_causal(test_set, eval_fn, model, causal_model, max_samples: int=None, causal_relations=None, iters: int=3):
    if max_samples is None:
        max_samples = len(test_set)

    all_accuracy = []
    all_rs = []
    all_ys = []

    for _ in range(iters):
        accuracy_list = []
        rs = []
        ys = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for _, sample in enumerate(test_set):
                future = executor.submit(eval_sample_causal, sample, eval_fn, model, causal_model, causal_relations)
                futures.append(future)
                if len(futures) >= max_samples:
                    break
            tqdm_loader = tqdm(concurrent.futures.as_completed(futures), total=len(futures), position=0)
            for future in tqdm_loader:
                try:
                    acc_item, response, y = future.result()
                    rs.append(response)
                    ys.append(y)
                    accuracy_list.append(acc_item)
                    tqdm_loader.set_description(f"Accuracy: {np.mean(accuracy_list):.4f}")
                except Exception as e:
                    print(f"Skipping a failed future: {e}")

        all_accuracy.extend(accuracy_list)
        all_rs.extend(rs)
        all_ys.extend(ys)

    return all_accuracy, all_rs, all_ys


def eval_dataset(test_set, eval_fn, model, max_samples: int=None, iters: int=3):
    if max_samples is None:
        max_samples = len(test_set)

    all_accuracy = []
    all_rs = []
    all_ys = []

    for _ in range(iters):
        accuracy_list = []
        rs = []
        ys = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for _, sample in enumerate(test_set):
                future = executor.submit(eval_sample, sample, eval_fn, model)
                futures.append(future)
                if len(futures) >= max_samples:
                    break
            tqdm_loader = tqdm(concurrent.futures.as_completed(futures), total=len(futures), position=0)
            for future in tqdm_loader:
                try:
                    acc_item, response, y = future.result()
                    rs.append(response)
                    ys.append(y)
                    accuracy_list.append(acc_item)
                    tqdm_loader.set_description(f"Accuracy: {np.mean(accuracy_list):.4f}")
                except Exception as e:
                    # print(f"Skipping a failed future: {e}")
                    pass

        all_accuracy.extend(accuracy_list)
        all_rs.extend(rs)
        all_ys.extend(ys)

    return all_accuracy, all_rs, all_ys

def eval_sample(item, eval_fn, model):
    """
    This function allows us to evaluate if an answer to a question in the prompt is a good answer.

    """
    x, y = item
    if isinstance(y, np.int64):
        y = int(y)
    x = tg.Variable(x, requires_grad=False, role_description="query to the language model")
    y = tg.Variable(y, requires_grad=False, role_description="correct answer for the query")
    response = model(x)
    try:
        eval_output_variable = eval_fn(inputs=dict(prediction=response, ground_truth_answer=y))
        return int(eval_output_variable.value), response.value, y.value
    except:
        eval_output_variable = eval_fn([x, y, response])
        eval_output_parsed = eval_fn.parse_output(eval_output_variable)
        return int(eval_output_parsed)