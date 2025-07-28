# round1a_logic.py (FINAL SIMPLIFIED & CORRECTED VERSION)

import os
import json
import pickle
import re
import fitz  # PyMuPDF
import numpy as np
from sklearn.preprocessing import StandardScaler

class BalancedPDFOutlineExtractor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_columns = [
            'font_size', 'font_size_ratio', 'font_above_avg', 'font_much_larger',
            'is_bold', 'is_italic', 'x_pos', 'y_pos', 'width', 'height',
            'word_count', 'char_count', 'line_count', 'is_multi_line', 'avg_words_per_line',
            'has_number', 'starts_with_number', 'is_all_caps', 'is_title_case', 
            'is_left_aligned', 'is_centered', 'is_top_of_page',
            'ends_with_colon', 'has_punctuation', 'is_isolated'
        ]

    def load_model(self, model_path='models/balanced_outline_extractor.pkl'):
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        print(f"Model loaded from {model_path}")

    def predict_outline(self, pdf_path):
        features = self._extract_text_features(pdf_path)
        if not features:
            return {"title": "", "outline": []}

        X = np.array([[f.get(col, 0) for col in self.feature_columns] for f in features])
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        title, headings = "", []
        for feature, pred in zip(features, predictions):
            if pred == 'title' and feature['page'] == 1 and not title:
                title = feature['text']
            elif pred in ['h1', 'h2', 'h3']:
                headings.append({'level': pred.upper(), 'text': feature['text'], 'page': feature['page'], 'y_pos': feature['y_pos']})
        
        if not title and features:
            title = features[0]['text']

        headings.sort(key=lambda x: (x['page'], x['y_pos']))
        
        # Remove y_pos before returning the final outline
        final_outline = [{'level': h['level'], 'text': h['text'], 'page': h['page']} for h in headings]
        
        return {"title": title, "outline": final_outline}

    def _extract_text_features(self, pdf_path):
        doc = fitz.open(pdf_path)
        features = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            grouped_spans = self._group_text_spans(blocks)
            for group in grouped_spans:
                text = group['text'].strip()
                if 3 <= len(text) <= 500:
                    feature = self._extract_group_features(group, text, page_num, page.rect)
                    if feature:
                        features.append(feature)
        self._add_document_features(features)
        doc.close()
        return features

    def _group_text_spans(self, blocks):
        all_spans = []
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        if span["text"].strip():
                            all_spans.append(span)
        all_spans.sort(key=lambda s: (s['bbox'][1], s['bbox'][0]))
        
        if not all_spans: return []
        groups, current_group = [], [all_spans[0]]
        for i in range(1, len(all_spans)):
            prev_span, curr_span = current_group[-1], all_spans[i]
            v_dist = curr_span['bbox'][1] - prev_span['bbox'][3]
            font_similar = abs(curr_span['size'] - prev_span['size']) < 1
            if v_dist < 5 and font_similar:
                current_group.append(curr_span)
            else:
                groups.append(self._finalize_group(current_group))
                current_group = [curr_span]
        groups.append(self._finalize_group(current_group))
        return groups

    def _finalize_group(self, spans):
        text = " ".join(s['text'] for s in spans).strip()
        text = re.sub(r'\s+', ' ', text)
        bbox = (min(s['bbox'][0] for s in spans), min(s['bbox'][1] for s in spans),
                max(s['bbox'][2] for s in spans), max(s['bbox'][3] for s in spans))
        return {'text': text, 'bbox': bbox, 'font_size': spans[0]['size'], 'font_flags': spans[0]['flags']}

    def _extract_group_features(self, group, text, page_num, page_rect):
        bbox, font_size, font_flags = group['bbox'], group['font_size'], group['font_flags']
        x_pos, y_pos = bbox[0] / page_rect.width, bbox[1] / page_rect.height
        width, height = (bbox[2] - bbox[0]) / page_rect.width, (bbox[3] - bbox[1]) / page_rect.height
        word_count = len(text.split())
        return {
            'text': text, 'page': page_num + 1, 'font_size': font_size,
            'is_bold': int(bool(font_flags & 2**4)), 'is_italic': int(bool(font_flags & 2**1)),
            'x_pos': x_pos, 'y_pos': y_pos, 'width': width, 'height': height,
            'word_count': word_count, 'char_count': len(text),
            'line_count': text.count('\n') + 1, 'is_multi_line': int(text.count('\n') > 0),
            'avg_words_per_line': word_count / (text.count('\n') + 1),
            'has_number': int(bool(re.search(r'\d', text))),
            'starts_with_number': int(text.lstrip().startswith(tuple('0123456789'))),
            'is_all_caps': int(text.isupper() and word_count > 1),
            'is_title_case': int(text.istitle()),
            'is_left_aligned': int(x_pos < 0.15), 'is_centered': int(0.3 < x_pos < 0.7),
            'is_top_of_page': int(y_pos < 0.15),
            'ends_with_colon': int(text.rstrip().endswith(':')),
            'has_punctuation': int(bool(re.search(r'[.!?]$', text))),
            'is_isolated': int(height > 20 and word_count < 10)
        }
    
    def _add_document_features(self, features):
        if not features: return
        font_sizes = [f['font_size'] for f in features]
        doc_max_font = max(font_sizes) if font_sizes else 0
        doc_avg_font = np.mean(font_sizes) if font_sizes else 0
        doc_std_font = np.std(font_sizes) if font_sizes else 0
        for f in features:
            f['font_size_ratio'] = f['font_size'] / doc_max_font if doc_max_font > 0 else 0
            f['font_above_avg'] = int(f['font_size'] > doc_avg_font + doc_std_font * 0.5)
            f['font_much_larger'] = int(f['font_size'] > doc_avg_font * 1.5)