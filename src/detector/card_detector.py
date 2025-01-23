import cv2
import numpy as np
from typing import List, Optional
from src.models.card import Card
from src.detector.template_matcher import TemplateMatcher
from src.detector.text_detector import TextDetector
from src.config.regions import *

class CardDetector:
    def __init__(self, template_matcher: TemplateMatcher):
        self.template_matcher = template_matcher
        self.text_detector = TextDetector()

    def detect_card(self, roi: np.ndarray, is_hero: bool = False) -> Optional[Card]:
        best_rank = None
        best_rank_conf = 0
        best_suit = None
        best_suit_conf = 0

        rank_templates = (self.template_matcher.hero_rank_templates 
                        if is_hero 
                        else self.template_matcher.community_rank_templates)
        
        suit_templates = (self.template_matcher.hero_suit_templates 
                         if is_hero 
                         else self.template_matcher.community_suit_templates)

        # Match rank
        for rank, template in rank_templates.items():
            conf, _ = self.template_matcher.match_template(roi, template)
            if conf > best_rank_conf:
                best_rank_conf = conf
                best_rank = rank

        # Match suit
        for suit, template in suit_templates.items():
            conf, _ = self.template_matcher.match_template(roi, template, 
                                                         use_preprocessing=False)
            if conf > best_suit_conf:
                best_suit_conf = conf
                best_suit = suit

        if best_rank_conf > 0.6 and best_suit_conf > 0.9:
            return Card(best_rank, best_suit, min(best_rank_conf, best_suit_conf))
        return None

    def detect_table_state(self, screen: np.ndarray):
        # Detect hero cards
        hero_cards = []
        for region in HERO_CARD_REGIONS:
            roi = screen[region['y1']:region['y2'], region['x1']:region['x2']]
            card = self.detect_card(roi, is_hero=True)
            if card:
                hero_cards.append(card)

        # Detect community cards
        community_cards = []
        for region in COMMUNITY_CARD_REGIONS:
            roi = screen[region['y1']:region['y2'], region['x1']:region['x2']]
            card = self.detect_card(roi, is_hero=False)
            if card:
                community_cards.append(card)

        # Detect stacks
        stacks = {}
        for player, region in STACK_REGIONS.items():
            roi = screen[region['y1']:region['y2'], region['x1']:region['x2']]
            stacks[player] = self.text_detector.detect_value(roi)

        # Detect bets
        bets = {}
        for player, region in BET_REGIONS.items():
            roi = screen[region['y1']:region['y2'], region['x1']:region['x2']]
            bets[player] = self.text_detector.detect_value(roi)

        # Detect pot
        pot_roi = screen[POT_REGION['y1']:POT_REGION['y2'], 
                        POT_REGION['x1']:POT_REGION['x2']]
        pot_size = self.text_detector.detect_value(pot_roi)

        return {
            'hero_cards': hero_cards,
            'community_cards': community_cards,
            'stacks': stacks,
            'bets': bets,
            'pot_size': pot_size
        }