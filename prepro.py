from ast import Constant
from tqdm import tqdm
import ujson as json
import numpy as np
import constant


def convert_token(token):
    if (token.lower() == '-lrb-'):
        return '('
    elif (token.lower() == '-rrb-'):
        return ')'
    elif (token.lower() == '-lsb-'):
        return '['
    elif (token.lower() == '-rsb-'):
        return ']'
    elif (token.lower() == '-lcb-'):
        return '{'
    elif (token.lower() == '-rcb-'):
        return '}'
    return token


class Processor:
    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.new_tokens = []
        self.tokenizer.add_tokens(self.new_tokens)
        if self.args.input_format not in ('typed_entity_marker'):
            raise Exception("Invalid input format!")

    def tokenize(self, tokens, subj_type, obj_type, ss, se, os, oe):
        sents = []
        input_format = self.args.input_format
        if input_format == 'typed_entity_marker':
            # typeMap = {"KNOW":"高等", "MAT":"初等", "GEOM":"几何", "PRIN":"定理", "":"初等"}
            # subj_start = '[{}]'.format(typeMap[subj_type])
            # subj_end = '[/{}]'.format(typeMap[subj_type])
            # obj_start = '[{}]'.format(typeMap[obj_type])
            # obj_end = '[/{}]'.format(typeMap[obj_type])

            # subj_start = '[SUBJ-{}]'.format(subj_type)
            # subj_end = '[/SUBJ-{}]'.format(subj_type)
            # obj_start = '[OBJ-{}]'.format(obj_type)
            # obj_end = '[/OBJ-{}]'.format(obj_type)
            
            subj_start = '#'
            subj_end = '#'
            obj_start = '$'
            obj_end = '$'
            for token in (subj_start, subj_end, obj_start, obj_end):
                if token not in self.new_tokens:
                    self.new_tokens.append(token)
                    self.tokenizer.add_tokens([token])

        for i_t, token in enumerate(tokens):
            tokens_wordpiece = self.tokenizer.tokenize(token)


            if input_format == 'typed_entity_marker':
                if i_t == ss:
                    new_ss = len(sents)
                    tokens_wordpiece = [subj_start] + tokens_wordpiece
                if i_t == se:
                    new_se = len(sents)
                    tokens_wordpiece = tokens_wordpiece + [subj_end]
                if i_t == os:
                    new_os = len(sents)
                    tokens_wordpiece = [obj_start] + tokens_wordpiece
                if i_t == oe:
                    new_oe = len(sents)
                    tokens_wordpiece = tokens_wordpiece + [obj_end]

            sents.extend(tokens_wordpiece)
        sents = sents[:self.args.max_seq_length - 2]
        input_ids = self.tokenizer.convert_tokens_to_ids(sents)
        input_ids = self.tokenizer.build_inputs_with_special_tokens(input_ids)
        return input_ids, new_ss + 1, new_os + 1

class MathProcessor(Processor):
    def __init__(self, args, tokenizer):
        super().__init__(args, tokenizer)
        # self.LABEL_TO_ID = {"依赖":  0, "被依赖": 1, "属于":  2, "包含":  3,
        #                     "反义":  4, "近义":  5, "属性":  6, "拥有":  7, "同位":  8, "无关":  9}
        if 'literature' in args.data_dir:
            print("dataset:literature")
            self.LABEL_TO_ID = {"unknown":  0, "Create": 1, "Use":  2, "Near":  3,
                            "Social":  4, "Located":  5, "Ownership":  6, "General-Special":  7, "Family":  8, "Part-Whole":  9}
        elif 'FinRE' in args.data_dir:
            print("dataset:FinRE")
            self.LABEL_TO_ID = {"unknown": 0,"注资": 1,"拥有": 2,"纠纷": 3,"自己": 4,"增持": 5,"重组": 6,"买资": 7,"签约": 8,"持股": 9,"交易": 10,"入股": 11,"转让": 12,"成立": 13,"分析": 14,"合作": 15,"帮助": 16,"发行": 17,"商讨": 18,"合并": 19,"竞争": 20,"订单": 21,"减持": 22,"合资": 23,"收购": 24,"借壳": 25,"欠款": 26,"被发行": 27,"被转让": 28,"被成立": 29,"被注资": 30,"被持股": 31,"被拥有": 32,"被收购": 33,"被帮助": 34,"被借壳": 35,"被买资": 36,"被欠款": 37,"被增持": 38,"拟收购": 39,"被减持": 40,"被分析": 41,"被入股": 42,"被拟收购": 43}

    def read(self, file_in):
        features = []
        with open(file_in, "r") as fh:
            data = json.load(fh)

        for d in tqdm(data):
            ss, se = d['subj_start'], d['subj_end']
            os, oe = d['obj_start'], d['obj_end']

            tokens = d['token']
            tokens = [convert_token(token) for token in tokens]

            input_ids, new_ss, new_os = self.tokenize(
                tokens, d['subj_type'], d['obj_type'], ss, se, os, oe)
            rel = self.LABEL_TO_ID[d['relation']]

            max_length = 180
            max_pos_length = 100

            p1, p2 = sorted((new_ss, new_os))
            _mask = np.zeros(max_length, dtype=np.long)
            _mask[p2 + 2: len(input_ids)] = 3
            _mask[p1 + 2: p2 + 2] = 2
            _mask[:p1 + 2] = 1
            _mask[len(input_ids):] = 0

            _pos1 = np.arange(max_length) - new_ss + max_pos_length
            _pos2 = np.arange(max_length) - new_os + max_pos_length
            _pos1[_pos1 > 2 * max_pos_length] = 2 * max_pos_length
            _pos1[_pos1 < 0] = 0
            _pos2[_pos2 > 2 * max_pos_length] = 2 * max_pos_length
            _pos2[_pos2 < 0] = 0

            feature = {
                'input_ids': input_ids,
                'labels': rel,
                'ss': new_ss,
                'os': new_os,
                'se': new_ss + se - ss,
                'oe': new_os + oe - os,
                "mask": _mask,
                "pos1": _pos1,
                "pos2": _pos2
            }

            features.append(feature)
        return features

