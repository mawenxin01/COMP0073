#!/usr/bin/env python3
"""
Medical Knowledge Base
Contains medical rules, definitions, and guidelines for emergency triage
"""

from typing import Dict, List, Tuple
import pandas as pd


class MedicalKnowledgeBase:
    """Medical Knowledge Base for Emergency Triage"""
    
    def __init__(self):
        """Initialize medical knowledge base"""
        
        # ESI (Emergency Severity Index) level definitions
        self.esi_definitions = {
            1: {
                "name": "Critical",
                "description": "Requires immediate resuscitation",
                "criteria": [
                    "Unstable vital signs",
                    "Requires immediate life-saving intervention",
                    "Unresponsive or altered mental status",
                    "Severe respiratory distress"
                ],
                "examples": [
                    "Cardiac arrest",
                    "Severe trauma with unstable vitals",
                    "Anaphylaxis",
                    "Severe respiratory failure"
                ]
            },
            2: {
                "name": "Emergent", 
                "description": "Requires urgent treatment within 10 minutes",
                "criteria": [
                    "High-risk situation",
                    "Severe pain or distress",
                    "Potential for deterioration",
                    "Requires rapid assessment"
                ],
                "examples": [
                    "Chest pain with concerning features",
                    "Severe abdominal pain",
                    "Altered mental status",
                    "Severe dehydration"
                ]
            },
            3: {
                "name": "Urgent",
                "description": "Requires treatment within 30 minutes", 
                "criteria": [
                    "Stable but concerning symptoms",
                    "Moderate pain or discomfort",
                    "Requires timely evaluation",
                    "Multiple complaints"
                ],
                "examples": [
                    "Moderate trauma",
                    "Fever with other symptoms",
                    "Moderate pain",
                    "Persistent vomiting"
                ]
            },
            4: {
                "name": "Less Urgent",
                "description": "Can wait 1-2 hours for treatment",
                "criteria": [
                    "Stable condition",
                    "Mild to moderate symptoms",
                    "Non-emergent complaints",
                    "Routine care needed"
                ],
                "examples": [
                    "Minor injuries",
                    "Cold symptoms",
                    "Mild pain",
                    "Routine follow-up"
                ]
            },
            5: {
                "name": "Non-urgent", 
                "description": "Can wait several hours or be referred",
                "criteria": [
                    "Stable chronic conditions",
                    "Minor complaints",
                    "Preventive care",
                    "Administrative issues"
                ],
                "examples": [
                    "Medication refills",
                    "Minor skin conditions", 
                    "Routine check-ups",
                    "Non-urgent referrals"
                ]
            }
        }
        
        # Critical symptom keywords that suggest higher acuity
        self.critical_keywords = [
            "chest pain", "difficulty breathing", "shortness of breath",
            "unconscious", "unresponsive", "cardiac arrest", "stroke",
            "severe bleeding", "trauma", "overdose", "anaphylaxis",
            "seizure", "severe abdominal pain", "altered mental status"
        ]
        
        # Vital sign normal ranges
        self.vital_ranges = {
            "heart_rate": {
                "normal": (60, 100),
                "concerning": [(50, 60), (100, 120)],
                "critical": [(0, 50), (120, 999)]
            },
            "systolic_bp": {
                "normal": (90, 140),
                "concerning": [(140, 160), (80, 90)],
                "critical": [(0, 80), (160, 999)]
            },
            "temperature": {
                "normal": (97.0, 99.5),
                "concerning": [(99.5, 101.0), (95.0, 97.0)],
                "critical": [(0, 95.0), (101.0, 999)]
            },
            "oxygen_saturation": {
                "normal": (95, 100),
                "concerning": [(90, 95)],
                "critical": [(0, 90)]
            }
        }
        
        # Time period risk factors
        self.time_period_factors = {
            0: {"name": "Night (0-6AM)", "risk_factor": 1.2, "note": "Limited staffing, patient may delay care"},
            1: {"name": "Morning (6AM-12PM)", "risk_factor": 1.0, "note": "Normal staffing levels"},
            2: {"name": "Afternoon (12PM-6PM)", "risk_factor": 1.1, "note": "Peak hours, higher volume"},
            3: {"name": "Evening (6PM-12AM)", "risk_factor": 1.3, "note": "Higher acuity, weekend effects"}
        }
    
    def get_esi_definition(self, level: int) -> Dict:
        """Get ESI level definition"""
        return self.esi_definitions.get(level, {})
    
    def assess_vital_signs(self, vitals: Dict) -> Dict:
        """Assess vital signs and return risk level"""
        
        assessment = {
            "overall_risk": "normal",
            "concerning_vitals": [],
            "critical_vitals": []
        }
        
        # Map input fields to standard names
        vital_mapping = {
            "heartrate": "heart_rate",
            "heart_rate": "heart_rate", 
            "sbp": "systolic_bp",
            "systolic_bp": "systolic_bp",
            "temperature": "temperature",
            "o2sat": "oxygen_saturation",
            "oxygen_saturation": "oxygen_saturation"
        }
        
        for input_key, standard_key in vital_mapping.items():
            if input_key in vitals and standard_key in self.vital_ranges:
                value = vitals[input_key]
                if value is None or pd.isna(value):
                    continue
                    
                ranges = self.vital_ranges[standard_key]
                
                # Check if critical
                for crit_range in ranges.get("critical", []):
                    if crit_range[0] <= value <= crit_range[1]:
                        assessment["critical_vitals"].append(f"{standard_key}: {value}")
                        assessment["overall_risk"] = "critical"
                        break
                
                # Check if concerning
                if assessment["overall_risk"] != "critical":
                    for conc_range in ranges.get("concerning", []):
                        if conc_range[0] <= value <= conc_range[1]:
                            assessment["concerning_vitals"].append(f"{standard_key}: {value}")
                            if assessment["overall_risk"] == "normal":
                                assessment["overall_risk"] = "concerning"
                            break
        
        return assessment
    
    def check_critical_symptoms(self, chief_complaint: str, keywords: str = None) -> List[str]:
        """Check for critical symptoms in chief complaint or keywords"""
        
        found_critical = []
        text_to_check = []
        
        if chief_complaint:
            text_to_check.append(chief_complaint.lower())
        if keywords:
            text_to_check.append(keywords.lower())
        
        combined_text = " ".join(text_to_check)
        
        for keyword in self.critical_keywords:
            if keyword.lower() in combined_text:
                found_critical.append(keyword)
        
        return found_critical
    
    def get_time_period_info(self, time_period: int) -> Dict:
        """Get time period information and risk factors"""
        return self.time_period_factors.get(time_period, {
            "name": "Unknown", 
            "risk_factor": 1.0, 
            "note": "Time period not specified"
        })
    
    def get_triage_recommendation(self, case_data: Dict) -> Dict:
        """Get comprehensive triage recommendation based on medical knowledge"""
        
        # Assess vital signs
        vital_assessment = self.assess_vital_signs(case_data)
        
        # Check for critical symptoms
        critical_symptoms = self.check_critical_symptoms(
            case_data.get("chiefcomplaint", ""),
            case_data.get("complaint_keywords", "")
        )
        
        # Get time period info
        time_period = case_data.get("time_period", 2)  # Default to afternoon
        time_info = self.get_time_period_info(time_period)
        
        # Determine recommended ESI level based on findings
        recommended_esi = 3  # Default to urgent
        reasoning = []
        
        # Critical vital signs or symptoms → ESI 1-2
        if vital_assessment["overall_risk"] == "critical" or critical_symptoms:
            if len(critical_symptoms) > 1 or vital_assessment["critical_vitals"]:
                recommended_esi = 1
                reasoning.append("Critical vital signs or multiple critical symptoms detected")
            else:
                recommended_esi = 2
                reasoning.append("Critical symptoms or concerning vital signs detected")
        
        # Concerning vital signs → ESI 2-3
        elif vital_assessment["overall_risk"] == "concerning":
            recommended_esi = 2
            reasoning.append("Concerning vital signs detected")
        
        # Age considerations
        age = case_data.get("age_at_visit", 50)
        if age and (age < 2 or age > 80):
            if recommended_esi > 2:
                recommended_esi = 2
                reasoning.append("Age-based risk factor (very young or elderly)")
        
        # Pain score considerations
        pain = case_data.get("pain", 0)
        if pain and pain >= 8:
            if recommended_esi > 2:
                recommended_esi = 2
                reasoning.append("Severe pain score")
        elif pain and pain >= 6:
            if recommended_esi > 3:
                recommended_esi = 3
                reasoning.append("Moderate pain score")
        
        return {
            "recommended_esi": recommended_esi,
            "reasoning": reasoning,
            "vital_assessment": vital_assessment,
            "critical_symptoms": critical_symptoms,
            "time_period_info": time_info,
            "esi_definition": self.get_esi_definition(recommended_esi)
        }