import numpy as np
import random

# 隨機加1個英文字母
def add_char_randomly(input_str , vocabulary = 'abcdefghijklmnopqrstuvwxy'):
    random_char = random.choice(list(vocabulary))
    i = random.randint(a = 0 , b = len(input_str) - 1)
    output_str = input_str[0 : i] + random_char + input_str[i:]
    assert len(output_str) == len(input_str) + 1
    assert random_char in output_str
    return output_str

# 隨機刪除1個字
def remove_char_randomly(input_str):
    i = random.randint(a = 0 , b = len(input_str) - 1)
    char_to_remove = input_str[i]
    before = input_str.count(char_to_remove)
    output_str = input_str[0 : i] + input_str[i + 1:]
    after = output_str.count(char_to_remove)
    assert before - 1 == after
    assert len(output_str) == len(input_str) - 1
    return output_str

# 隨機選取1個字與英文字母調換
def change_char_randomly(input_str , vocabulary = 'abcdefghijklmnopqrstuvwxy'):
    random_char = random.choice(list(vocabulary))
    i = random.randint(a = 0 , b = len(input_str) - 1)
    output_str = input_str[0 : i] + random_char + input_str[i + 1:]
    assert len(output_str) == len(input_str)
    assert random_char in output_str
    return output_str

# 隨機選取2個字互相交換
def permute_two_chars_randomly(input_str):
    input_str = list(input_str)
    i1 = random.randint(a = 0 , b = len(input_str) - 1)
    i2 = random.randint(a = 0 , b = len(input_str) - 1)
    temp = input_str[i1]
    input_str[i1] = input_str[i2]
    input_str[i2] = temp
    output_str = ''.join(input_str)
    return output_str


def data_augmentation_function_set():
    return [add_char_randomly , remove_char_randomly , change_char_randomly , permute_two_chars_randomly]
