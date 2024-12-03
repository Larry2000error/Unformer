import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import random
from nltk.corpus import wordnet
import re


def int_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval .

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.

  Returns:
    An int that results from scaling `maxval` according to `level`.
  """
  return int(level * maxval / 10)


def float_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval.

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.

  Returns:
    A float that results from scaling `maxval` according to `level`.
  """

  return float(level) * maxval / 10.
def sample_level(n):
  return np.random.uniform(low=0.1, high=n)

def autocontrast(pil_img, _):
  return ImageOps.autocontrast(pil_img)


def equalize(pil_img, _):
  return ImageOps.equalize(pil_img)


def posterize(pil_img, level):
  level = int_parameter(sample_level(level), 4)
  return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level):
  degrees = int_parameter(sample_level(level), 30)
  if np.random.uniform() > 0.5:
    degrees = -degrees
  return pil_img.rotate(degrees, resample=Image.BILINEAR)


def solarize(pil_img, level):
  level = int_parameter(sample_level(level), 256)
  return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level):
  level = float_parameter(sample_level(level), 0.3)
  if np.random.uniform() > 0.5:
    level = -level
  return pil_img.transform(pil_img.size,
                           Image.AFFINE, (1, level, 0, 0, 1, 0),
                           resample=Image.BILINEAR)


def shear_y(pil_img, level):
  level = float_parameter(sample_level(level), 0.3)
  if np.random.uniform() > 0.5:
    level = -level
  return pil_img.transform(pil_img.size,
                           Image.AFFINE, (1, 0, 0, level, 1, 0),
                           resample=Image.BILINEAR)


def translate_x(pil_img, level):
  level = int_parameter(sample_level(level), pil_img.size[0] / 3)
  if np.random.random() > 0.5:
    level = -level
  return pil_img.transform(pil_img.size,
                           Image.AFFINE, (1, 0, level, 0, 1, 0),
                           resample=Image.BILINEAR)


def translate_y(pil_img, level):
  level = int_parameter(sample_level(level), pil_img.size[1] / 3)
  if np.random.random() > 0.5:
    level = -level
  return pil_img.transform(pil_img.size,
                           Image.AFFINE, (1, 0, 0, 0, 1, level),
                           resample=Image.BILINEAR)


# operation that overlaps with ImageNet-C's test set
def color(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def contrast(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def brightness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def sharpness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)

img_augmentations = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y, color, contrast, brightness, sharpness
]


# def synonym_replacement(text):
#     if isinstance(text, str):
#         words = text.split()
#     else:
#         words = text
#
#     new_words = words.copy()
#
#     for i, word in enumerate(words):
#         synonyms = wordnet.synsets(word)
#         if synonyms:
#             # 获取所有同义词
#             all_synonyms = set()
#             for syn in synonyms:
#                 for lemma in syn.lemmas():
#                     # 只选择常见的同义词
#                     if lemma.name() != word and lemma.count() > 1:
#                         all_synonyms.add(lemma.name())
#             # 移除原词
#             all_synonyms.discard(word)
#
#             # 如果有可用的同义词，选择其中一个
#             if all_synonyms:
#                 synonym = random.choice(list(all_synonyms))
#                 # 检查是否与上下文合适（简化示例，实际应用可更复杂）
#                 if synonym.isalpha():  # 确保是一个有效的词
#                     new_words[i] = synonym
def synonym_replacement(text):
    # 使用正则表达式分隔标记和其他文本
    parts = re.split(r'(<o>.*?</o>|<s>.*?</s>)', text)
    new_parts = []

    for part in parts:
        if part.startswith('<o>') and part.endswith('</o>'):
            # 不替换<o>标记之间的内容
            new_parts.append(part)
        elif part.startswith('<s>') and part.endswith('</s>'):
            # 不替换<s>标记之间的内容
            new_parts.append(part)
        else:
            words = part.split()
            new_words = words.copy()
            for i, word in enumerate(words):
                synonyms = wordnet.synsets(word)
                if synonyms:
                    all_synonyms = set()
                    for syn in synonyms:
                        for lemma in syn.lemmas():
                            if lemma.name() != word and lemma.count() > 1:
                                all_synonyms.add(lemma.name())
                    all_synonyms.discard(word)

                    if all_synonyms:
                        synonym = random.choice(list(all_synonyms))
                        if synonym.isalpha():
                            new_words[i] = synonym
            new_parts.append(' '.join(new_words))

    return ''.join(new_parts)


def random_insertion(text, n=1):
    if type(text) is str:
        words = text.split()
    else:
        words = text
    for _ in range(n):
        new_word = random.choice(words)
        random_index = random.randint(0, len(words) - 1)
        words.insert(random_index, new_word)
    return words


# from googletrans import Translator
#
# translator = Translator()
#
# def back_translation(text):
#     # 原始文本翻译为另一种语言
#     translated = translator.translate(text, dest='fr').text  # 法语
#     # 然后再翻译回原语言
#     back_translated = translator.translate(translated, dest='en').text  # 英语
#     return back_translated


text_augmentations = [synonym_replacement] #, random_insertion] #, back_translation]

