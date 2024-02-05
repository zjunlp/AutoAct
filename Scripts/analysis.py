import jsonlines
import argparse

def analyze_results(file_path):
    all_correct = 0
    all_wrong = 0
    right_reflect_wrong = 0
    wrong_reflect_wrong = 0
    reward = 0
    all_row = 0
    with open(file_path, "r") as f:
        for item in jsonlines.Reader(f):
            all_row += 1
            reward += item["reward"]
            if item["correct"]:
                all_correct += 1
                if "Reflect[wrong]" in item["prompt"]:
                    right_reflect_wrong += 1
            else:
                all_wrong += 1
                if "Reflect[wrong]" in item["prompt"]:
                    wrong_reflect_wrong += 1
    print(f'1. Accuracy: {all_correct/all_row}')
    print(f'2. Reflect wrong: {(right_reflect_wrong+wrong_reflect_wrong)/all_row}  Reflect right: {1-(right_reflect_wrong+wrong_reflect_wrong)/all_row}')
    print(f'3. Correct answers - Reflect right: {1-(right_reflect_wrong/all_correct)}  Wrong: {right_reflect_wrong/all_correct}')
    print(f'4. Wrong answers - Reflect right: {1-(wrong_reflect_wrong/all_wrong)}  Wrong: {wrong_reflect_wrong/all_wrong}')
    print(f'5. Reward: {reward}')


def main():
    parser = argparse.ArgumentParser(description="Analyze results from JSONL file.")
    parser.add_argument("--file_path", help="Path to the directory containing JSONL files.")
    args = parser.parse_args()
    analyze_results(args.file_path)


if __name__ == "__main__":
    main()
