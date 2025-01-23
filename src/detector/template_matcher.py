import cv2
import numpy as np
import os
from typing import Dict, Tuple

class TemplateMatcher:
    def __init__(self, template_path: str):
        self.template_path = template_path
        self.hero_rank_templates = {}
        self.hero_suit_templates = {}
        self.community_rank_templates = {}
        self.community_suit_templates = {}
        self.load_templates()

    def load_templates(self):
        """Load all template images from the template directory"""
        # Load hero rank templates
        self._load_templates('ranks_hero', self.hero_rank_templates)
        self._load_templates('suits_hero', self.hero_suit_templates)
        self._load_templates('ranks_community', self.community_rank_templates)
        self._load_templates('suits_community', self.community_suit_templates)

    def _load_templates(self, subfolder: str, template_dict: Dict):
        path = os.path.join(self.template_path, subfolder)
        for filename in os.listdir(path):
            if filename.endswith('.png'):
                key = filename.split('.')[0]
                template = cv2.imread(os.path.join(path, filename))
                if template is not None:
                    template_dict[key] = template

    def match_template(self, image: np.ndarray, template: np.ndarray, 
                      use_preprocessing: bool = True) -> Tuple[float, Tuple[int, int]]:
        if use_preprocessing:
            processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            processed_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            processed_image = image
            processed_template = template

        result = cv2.matchTemplate(processed_image, processed_template, 
                                 cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        return max_val, max_loc