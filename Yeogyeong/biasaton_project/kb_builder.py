# kb_builder.py

import os
import re
import json
import pandas as pd
from typing import List, Dict, Any, Optional


class BiasKnowledgeBase:
    def __init__(self):
        self.bias_patterns = {}
        self.bias_counters = {}
        self.nationality_bias = {}
        self.occupation_bias = {}
        self.gender_bias = {}
        self.religion_bias = {}
        self.disability_bias = {}
        self.region_bias = {}
        self.age_bias = {}

    def build_general_kb(self):
        """
        평가 데이터에 의존하지 않고 일반적인 편향 유형에 기반한 KB 구축
        """
        self._build_predefined_bias_patterns()
        self._build_predefined_nationality_bias()
        self._build_predefined_occupation_bias()
        self._build_predefined_gender_bias()
        self._build_predefined_religion_bias()
        self._build_predefined_disability_bias()
        self._build_predefined_region_bias()
        self._build_predefined_age_bias()
        self._build_countering_strategies()

        return self

    def _build_predefined_bias_patterns(self):
        """사전 정의된 일반적인 편향 패턴 구축"""
        self.bias_patterns = {
            "pattern_nationality": {
                "keywords": ["국적", "인종", "민족", "출신", "외국인", "이민자"],
                "description": "국적/인종에 기반한 고정관념과 편향",
                "examples": [
                    "특정 국가 출신이라는 이유로 능력, 성격, 행동을 가정하는 경우",
                    "선진국 vs 개발도상국 출신자를 대비시켜 우열을 가정하는 경우",
                    "특정 민족에 대한 고정관념을 기반으로 판단하는 경우"
                ]
            },
            "pattern_gender": {
                "keywords": ["성별", "남성", "여성", "결혼", "가사", "육아"],
                "description": "성별에 기반한 고정관념과 편향",
                "examples": [
                    "성별에 따라 특정 역할이나 행동을 기대하는 경우",
                    "결혼 관계에서 성별에 따라 책임과 권한을 다르게 보는 경우",
                    "직업적 역량을 성별과 연관 짓는 경우"
                ]
            },
            "pattern_occupation": {
                "keywords": ["직업", "학력", "소득", "계층", "지위"],
                "description": "직업/학력에 기반한 고정관념과 편향",
                "examples": [
                    "특정 직업군의 사람들에게 지능이나 능력을 가정하는 경우",
                    "학력에 따라 인성이나 가치관을 판단하는 경우",
                    "경제적 지위에 따라 도덕성이나 정직성을 연관 짓는 경우"
                ]
            },
            "pattern_religion": {
                "keywords": ["종교", "신앙", "믿음", "교회", "사원", "무신론"],
                "description": "종교에 기반한 고정관념과 편향",
                "examples": [
                    "특정 종교인에 대한 가치관이나 도덕성을 가정하는 경우",
                    "종교적 배경에 따라 개방성이나 관용을 판단하는 경우",
                    "무신론자와 종교인의 윤리관을 대비시키는 경우"
                ]
            },
            "pattern_disability": {
                "keywords": ["장애", "질환", "휠체어", "시각장애", "청각장애"],
                "description": "장애에 기반한 고정관념과 편향",
                "examples": [
                    "장애 여부에 따라 능력이나 독립성을 가정하는 경우",
                    "장애인의 업무 수행 능력을 일반화하는 경우",
                    "장애와 성격 특성을 연관 짓는 경우"
                ]
            },
            "pattern_region": {
                "keywords": ["지역", "출신", "서울", "지방", "도시", "농촌"],
                "description": "출신 지역에 기반한 고정관념과 편향",
                "examples": [
                    "특정 지역 출신자의 성격이나 행동을 일반화하는 경우",
                    "지역에 따라 교육 수준이나 문화적 소양을 가정하는 경우",
                    "도시와 농촌 출신을 대비시켜 우열을 판단하는 경우"
                ]
            },
            "pattern_age": {
                "keywords": ["나이", "세대", "젊은", "노인", "청년", "장년"],
                "description": "나이/세대에 기반한 고정관념과 편향",
                "examples": [
                    "특정 연령대의 기술 활용 능력을 가정하는 경우",
                    "세대에 따라 책임감이나 성실성을 판단하는 경우",
                    "나이와 창의성이나 적응력을 연관 짓는 경우"
                ]
            }
        }

    def _build_predefined_nationality_bias(self):
        """사전 정의된 국적 관련 편향 정보 구축"""
        nationalities = [
            "베트남인", "말레이시아인", "덴마크인", "태국인", "노르웨이인", "가나인",
            "스웨덴인", "방글라데시인", "피지인", "벨라루스인", "우즈베키스탄인",
            "사모아인", "독일인", "핀란드인", "튀니지인", "남아프리카공화국인", "대만인",
            "프랑스인", "파키스탄인", "인도인", "미국인", "라오스인", "몽골인", "호주인"
        ]

        for nationality in nationalities:
            self.nationality_bias[nationality] = {
                "frequency": 0,  # 실제 데이터 없이 기본값 설정
                "positive_associations": {},
                "negative_associations": {},
                "neutral_framing": f"{nationality}의 행동과 특성은 국적이 아닌 개인의 실제 행동, 발언, 상황적 맥락에 기반하여 판단해야 합니다."
            }

    def _build_predefined_occupation_bias(self):
        """사전 정의된 직업 관련 편향 정보 구축"""
        occupations = [
            "변호사", "의사", "교수", "CEO", "임원", "경비원", "청소부",
            "배달원", "노동자", "종업원", "직원", "학생", "공장 노동자",
            "IT 기업 CEO", "사무원", "간호사", "엔지니어", "교사"
        ]

        for occupation in occupations:
            self.occupation_bias[occupation] = {
                "frequency": 0,  # 실제 데이터 없이 기본값 설정
                "positive_associations": {},
                "negative_associations": {},
                "neutral_framing": f"{occupation}의 능력과 특성은 직업 자체가 아닌 개인의 행동, 성과, 구체적 상황에 기반하여 판단해야 합니다."
            }

    def _build_predefined_gender_bias(self):
        """사전 정의된 성별 관련 편향 정보 구축"""
        gender_terms = {
            "남성": ["남성", "남자", "남편"],
            "여성": ["여성", "여자", "아내"],
            "트랜스젠더": ["트랜스젠더", "젠더퀴어"]
        }

        for gender, terms in gender_terms.items():
            self.gender_bias[gender] = {
                "frequency": 0,  # 실제 데이터 없이 기본값 설정
                "positive_associations": {},
                "negative_associations": {},
                "neutral_framing": f"{gender}의 행동과 특성은 성별 고정관념이 아닌 개인의 실제 행동, 가치관, 선택에 기반하여 판단해야 합니다."
            }

    def _build_predefined_religion_bias(self):
        """사전 정의된 종교 관련 편향 정보 구축"""
        religions = ["기독교", "불교", "이슬람교", "힌두교", "무신론", "유대교", "신흥종교"]

        for religion in religions:
            self.religion_bias[religion] = {
                "frequency": 0,  # 실제 데이터 없이 기본값 설정
                "positive_associations": {},
                "negative_associations": {},
                "neutral_framing": f"{religion}를 믿는 사람의 행동과 가치관은 종교 자체가 아닌 개인의 실제 행동, 발언, 선택에 기반하여 판단해야 합니다."
            }

    def _build_predefined_disability_bias(self):
        """사전 정의된 장애 관련 편향 정보 구축"""
        disabilities = ["시각장애", "청각장애", "지체장애", "발달장애", "정신장애"]

        for disability in disabilities:
            self.disability_bias[disability] = {
                "frequency": 0,  # 실제 데이터 없이 기본값 설정
                "positive_associations": {},
                "negative_associations": {},
                "neutral_framing": f"{disability}가 있는 사람의 능력과 특성은 장애 여부가 아닌 개인의 실제 행동, 성취, 상황적 맥락에 기반하여 판단해야 합니다."
            }

    def _build_predefined_region_bias(self):
        """사전 정의된 지역 관련 편향 정보 구축"""
        regions = ["서울", "경기", "강원", "충청", "전라", "경상", "제주"]

        for region in regions:
            self.region_bias[region] = {
                "frequency": 0,  # 실제 데이터 없이 기본값 설정
                "positive_associations": {},
                "negative_associations": {},
                "neutral_framing": f"{region} 출신 사람의 성격과 행동은 지역적 배경이 아닌 개인의 실제 행동, 발언, 가치관에 기반하여 판단해야 합니다."
            }

    def _build_predefined_age_bias(self):
        """사전 정의된 나이 관련 편향 정보 구축"""
        age_groups = ["청소년", "청년", "중년", "노년", "신입", "장년층"]

        for age in age_groups:
            self.age_bias[age] = {
                "frequency": 0,  # 실제 데이터 없이 기본값 설정
                "positive_associations": {},
                "negative_associations": {},
                "neutral_framing": f"{age}의 능력과 특성은 나이 자체가 아닌 개인의 실제 행동, 경험, 상황적 맥락에 기반하여 판단해야 합니다."
            }

    def _build_countering_strategies(self):
        """편향 대응 전략 구축"""
        self.bias_counters = {
            "nationality": {
                "detection": "국적 기반 가정이 질문에 있는지 확인 (예: 특정 국적 사람의 행동에 대한 가정)",
                "strategy": "국적 대신 명시적으로 언급된 행동과 특성에만 집중",
                "examples": [
                    {"biased": "베트남인이니까 시끄럽고 무례할 것이다",
                     "neutral": "시끄럽게 대화하는 것은 국적이 아닌 현재 상황(오랜만에 친구 만남)과 개인적 선택에 따른 것이다"},
                    {"biased": "유럽인이니까 더 예의바르고 교양있을 것이다",
                     "neutral": "예의와 교양은 국적보다 개인의 성장 배경, 교육, 가치관 등 다양한 요소에 영향을 받는다"}
                ]
            },
            "occupation": {
                "detection": "직업과 능력/특성을 직접 연결하는 가정이 있는지 확인",
                "strategy": "직업보다 실제 행동과 결과에 기반한 판단",
                "examples": [
                    {"biased": "CEO니까 더 똑똑하고 능력있을 것이다",
                     "neutral": "직업적 지위와 지능/능력은 다른 개념이며, 다양한 요소에 영향을 받는다"},
                    {"biased": "노동자니까 교양이나 지식이 부족할 것이다",
                     "neutral": "직업과 교양/지식은 독립적이며 다양한 경험과 학습을 통해 얻어질 수 있다"}
                ]
            },
            "gender": {
                "detection": "성별에 따른 역할과 특성에 대한 고정관념이 있는지 확인",
                "strategy": "성별과 무관하게 개인의 선택과 행동에 집중",
                "examples": [
                    {"biased": "남자니까 집안일에 무관심할 것이다",
                     "neutral": "집안일에 대한 관심과 참여는 성별이 아닌 개인의 가치관과 가정 내 역할 분담에 영향을 받는다"},
                    {"biased": "여자니까 리더십이 부족할 것이다",
                     "neutral": "리더십은 성별과 무관하게 개인의 경험, 교육, 성격에 따라 다양하게 나타날 수 있다"}
                ]
            },
            "religion": {
                "detection": "종교와 가치관/행동을 자동 연결하는 가정이 있는지 확인",
                "strategy": "종교적 배경보다 명시적 행동과 발언에 집중",
                "examples": [
                    {"biased": "무신론자니까 도덕성이 부족할 것이다",
                     "neutral": "도덕적 가치관은 종교적 신념과 별개로 다양한 경험과 사회적 학습을 통해 형성된다"},
                    {"biased": "종교인이니까 보수적일 것이다",
                     "neutral": "정치적 성향은 종교와 독립적으로 다양한 요소에 영향을 받아 형성된다"}
                ]
            },
            "disability": {
                "detection": "장애와 능력을 연결하는 가정이 있는지 확인",
                "strategy": "장애 자체보다 실제 행동과 성취에 집중",
                "examples": [
                    {"biased": "시각장애인이니까 이해력이 떨어질 것이다",
                     "neutral": "이해력은 시각적 정보 접근성과 별개의 인지 능력이며 다양한 방식으로 발현될 수 있다"},
                    {"biased": "장애인이니까 의존적일 것이다",
                     "neutral": "독립성은 장애 여부보다 개인의 성향, 환경적 지원, 적응 능력 등 다양한 요소에 영향을 받는다"}
                ]
            },
            "region": {
                "detection": "지역과 성격/특성을 연결하는 가정이 있는지 확인",
                "strategy": "지역 출신보다 개인적 행동과 특성에 집중",
                "examples": [
                    {"biased": "서울 사람이니까 세련되고 빠를 것이다",
                     "neutral": "세련됨과 생활 템포는 지역보다 개인의 경험, 환경, 선호도에 영향을 받는다"},
                    {"biased": "시골 출신이니까 순박할 것이다",
                     "neutral": "성격은 출신 지역보다 개인의 성장 환경, 경험, 유전적 요소 등에 더 크게 영향을 받는다"}
                ]
            },
            "age": {
                "detection": "나이와 능력/특성을 연결하는 가정이 있는지 확인",
                "strategy": "나이보다 개인의 행동과 태도에 집중",
                "examples": [
                    {"biased": "노인이니까 기술을 못 다룰 것이다",
                     "neutral": "기술 활용 능력은 나이보다 개인의 경험, 학습 의지, 접근성 등에 영향을 받는다"},
                    {"biased": "젊은이니까 책임감이 부족할 것이다",
                     "neutral": "책임감은 나이보다 개인의 성장 환경, 가치관, 경험에 더 크게 영향을 받는다"}
                ]
            }
        }

    def retrieve_for_query(self, query_text):
        """쿼리에 관련된 KB 항목 검색"""
        result = {
            "identified_biases": [],
            "relevant_counters": [],
            "neutral_frames": []
        }

        # 패턴 기반 편향 유형 식별
        for pattern_name, pattern_info in self.bias_patterns.items():
            if any(keyword in query_text.lower() for keyword in pattern_info["keywords"]):
                result["identified_biases"].append({
                    "type": pattern_name,
                    "description": pattern_info["description"],
                    "examples": pattern_info["examples"]
                })

        # 국적 관련 편향 확인
        for nationality in self.nationality_bias:
            if nationality in query_text:
                result["identified_biases"].append({
                    "type": "nationality",
                    "entity": nationality,
                    "data": self.nationality_bias[nationality]
                })
                result["neutral_frames"].append(self.nationality_bias[nationality]["neutral_framing"])
                if "nationality" in self.bias_counters:
                    result["relevant_counters"].append(self.bias_counters["nationality"])

        # 직업 관련 편향 확인
        for occupation in self.occupation_bias:
            if occupation in query_text:
                result["identified_biases"].append({
                    "type": "occupation",
                    "entity": occupation,
                    "data": self.occupation_bias[occupation]
                })
                result["neutral_frames"].append(self.occupation_bias[occupation]["neutral_framing"])
                if "occupation" in self.bias_counters:
                    result["relevant_counters"].append(self.bias_counters["occupation"])

        # 성별 관련 편향 확인
        for gender, terms in {"남성": ["남성", "남자", "남편"], "여성": ["여성", "여자", "아내"], "트랜스젠더": ["트랜스젠더", "젠더퀴어"]}.items():
            if any(term in query_text for term in terms):
                result["identified_biases"].append({
                    "type": "gender",
                    "entity": gender,
                    "data": self.gender_bias.get(gender, {})
                })
                bias_info = self.gender_bias.get(gender, {})
                if "neutral_framing" in bias_info:
                    result["neutral_frames"].append(bias_info["neutral_framing"])
                if "gender" in self.bias_counters:
                    result["relevant_counters"].append(self.bias_counters["gender"])

        # 추가 편향 검사 (종교, 장애, 지역, 연령 등)
        for bias_type, bias_dict, counter_key in [
            ("religion", self.religion_bias, "religion"),
            ("disability", self.disability_bias, "disability"),
            ("region", self.region_bias, "region"),
            ("age", self.age_bias, "age")
        ]:
            for entity, info in bias_dict.items():
                if entity in query_text:
                    result["identified_biases"].append({
                        "type": bias_type,
                        "entity": entity,
                        "data": info
                    })
                    if "neutral_framing" in info:
                        result["neutral_frames"].append(info["neutral_framing"])
                    if counter_key in self.bias_counters:
                        result["relevant_counters"].append(self.bias_counters[counter_key])

        return result

    def save_to_json(self, output_path):
        """KB를 JSON 파일로 저장"""
        data = {
            "bias_patterns": self.bias_patterns,
            "nationality_bias": self.nationality_bias,
            "occupation_bias": self.occupation_bias,
            "gender_bias": self.gender_bias,
            "religion_bias": self.religion_bias,
            "disability_bias": self.disability_bias,
            "region_bias": self.region_bias,
            "age_bias": self.age_bias,
            "bias_counters": self.bias_counters
        }

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✅ KB 저장 완료: {output_path}")

    def _preprocess_text(self, text):
        """텍스트 전처리: 특수문자 제거 및 소문자 변환"""
        text = re.sub(r"[^\w\s]", " ", text)  # 특수문자 제거
        text = re.sub(r"\s+", " ", text)  # 공백 정리
        return text.lower().strip()


# 실행 예시
if __name__ == "__main__":
    kb = BiasKnowledgeBase()
    # 평가 데이터를 사용하지 않고 일반적인 편향 유형에 기반한 KB 구축
    kb.build_general_kb()
    kb.save_to_json("../KB/bias_knowledge_base.json")