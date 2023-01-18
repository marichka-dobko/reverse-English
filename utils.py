from datasets import load_dataset


def main():
    # Calculating the maximum number of words per article in our dataset
    dataset = load_dataset("ag_news")

    largest_len = 0
    for i in range(len(dataset['train'])):
        if len(dataset['train'][i]['text'].split()) > largest_len:
            largest_len = len(dataset['train'][i]['text'].split())

    for i in range(len(dataset['test'])):
        if len(dataset['test'][i]['text'].split()) > largest_len:
            largest_len = len(dataset['test'][i]['text'].split())
    print(largest_len)


if __name__ == '__main__':
    main()
