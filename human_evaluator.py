
import json
from latent_dialog.evaluators import MultiWozEvaluator, CamRestEvaluator


if __name__ == "__main__": 

    # Human Evaluation
    data_name = "multiwoz_2.0"
    # data_name = "multiwoz_2.1"
    evaluator = MultiWozEvaluator(data_name)

    # data_name = "camrest"
    # evaluator = CamRestEvaluator(data_name)

    with open("data/%s/test_dials.json"%data_name, "r") as f:
        human_raw_data = json.load(f)

    generated_data = {}
    for key, value in human_raw_data.items():
        generated_data[key] = value["sys"]

    r = evaluator.evaluateModel(generated_data, False, mode="valid")
    print(r[0])



