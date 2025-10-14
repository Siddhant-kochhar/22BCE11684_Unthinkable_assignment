"""
Gemini AI Helper for generating product recommendation explanations
"""

import google.generativeai as genai
import os
from typing import Dict, Optional
import json


class GeminiExplainer:
    """Helper class for generating AI-powered recommendation explanations"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini AI client
        
        Args:
            api_key: Gemini API key. If None, will try to get from environment
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key not provided. Set GEMINI_API_KEY environment variable or pass api_key parameter.")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')  # Updated model name
    
    def get_recommendation_explanation(self, 
                                    target_product: Dict, 
                                    recommended_product: Dict, 
                                    recommendation_type: str = "content-based") -> str:
        """
        Generate explanation for why a product is recommended
        
        Args:
            target_product: The product user is interested in
            recommended_product: The recommended product
            recommendation_type: Type of recommendation (content-based, collaborative, hybrid)
            
        Returns:
            AI-generated explanation string
        """
        try:
            prompt = self._create_explanation_prompt(target_product, recommended_product, recommendation_type)
            
            response = self.model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            print(f"Error generating explanation: {e}")
            return f"This product is recommended based on similarity to '{target_product.get('Name', 'your selection')}'."
    
    def get_product_summary(self, product: Dict) -> str:
        """
        Generate a brief product summary
        
        Args:
            product: Product dictionary
            
        Returns:
            AI-generated product summary
        """
        try:
            prompt = f"""
            Create a brief, engaging product summary for this e-commerce item:
            
            Product Name: {product.get('Name', 'N/A')}
            Brand: {product.get('Brand', 'N/A')}
            Rating: {product.get('Rating', 'N/A')}/5
            Review Count: {product.get('ReviewCount', 'N/A')} reviews
            
            Write a 2-3 sentence summary that highlights key features and benefits.
            Keep it concise and customer-focused.
            """
            
            response = self.model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            print(f"Error generating product summary: {e}")
            return f"High-quality {product.get('Brand', '')} product with {product.get('Rating', 'great')} rating."
    
    def get_trending_explanation(self, products: list) -> str:
        """
        Generate explanation for trending products
        
        Args:
            products: List of trending product dictionaries
            
        Returns:
            AI-generated explanation for trending status
        """
        try:
            top_brands = list(set([p.get('Brand', '') for p in products[:5] if p.get('Brand')]))[:3]
            avg_rating = sum([p.get('Rating', 0) for p in products[:5]]) / min(5, len(products))
            
            prompt = f"""
            Explain why these products are trending in e-commerce:
            
            Top trending brands: {', '.join(top_brands)}
            Average rating: {avg_rating:.1f}/5
            Number of products: {len(products)}
            
            Write a brief 2-3 sentence explanation of what makes these products popular.
            Focus on quality, customer satisfaction, and market trends.
            """
            
            response = self.model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            print(f"Error generating trending explanation: {e}")
            return "These products are trending due to high customer ratings and strong market demand."
    
    def _create_explanation_prompt(self, target_product: Dict, recommended_product: Dict, rec_type: str) -> str:
        """Create detailed prompt for recommendation explanation"""
        
        base_prompt = f"""
        You are an AI shopping assistant. Explain why we're recommending this product to a customer.
        
        Customer was looking at:
        - Product: {target_product.get('Name', 'N/A')}
        - Brand: {target_product.get('Brand', 'N/A')}
        - Rating: {target_product.get('Rating', 'N/A')}/5
        
        We're recommending:
        - Product: {recommended_product.get('Name', 'N/A')}
        - Brand: {recommended_product.get('Brand', 'N/A')}
        - Rating: {recommended_product.get('Rating', 'N/A')}/5
        - Reviews: {recommended_product.get('ReviewCount', 'N/A')} customer reviews
        
        Recommendation method: {rec_type}
        """
        
        if rec_type == "content-based":
            prompt = base_prompt + """
            
            This recommendation is based on product similarity (features, brand, category).
            Write a friendly 2-3 sentence explanation focusing on:
            - Why these products are similar
            - What benefits the customer will get
            - Why this is a good match for their interests
            
            Keep it conversational and helpful, like a knowledgeable store assistant.
            """
        
        elif rec_type == "collaborative":
            prompt = base_prompt + """
            
            This recommendation is based on what similar customers also liked.
            Write a friendly 2-3 sentence explanation focusing on:
            - How other customers with similar tastes loved this product
            - Why it's a popular choice among people with similar preferences
            - What makes it a great complementary purchase
            
            Keep it conversational and emphasize social proof.
            """
        
        else:  # hybrid or other
            prompt = base_prompt + """
            
            This recommendation combines multiple factors: product similarity and customer behavior.
            Write a friendly 2-3 sentence explanation focusing on:
            - Why this is an intelligent match
            - How it combines the best of both similarity and popularity
            - What value it brings to the customer
            
            Keep it conversational and confidence-inspiring.
            """
        
        return prompt


# Global instance for easy import
gemini_explainer = None

def get_gemini_explainer() -> GeminiExplainer:
    """Get or create global Gemini explainer instance"""
    global gemini_explainer
    if gemini_explainer is None:
        gemini_explainer = GeminiExplainer()
    return gemini_explainer

def get_explanation(target_product: Dict, recommended_product: Dict, rec_type: str = "content-based") -> str:
    """
    Convenience function to get recommendation explanation
    
    Args:
        target_product: Product user was interested in
        recommended_product: Recommended product
        rec_type: Type of recommendation
        
    Returns:
        AI-generated explanation
    """
    try:
        explainer = get_gemini_explainer()
        return explainer.get_recommendation_explanation(target_product, recommended_product, rec_type)
    except Exception as e:
        print(f"Error getting explanation: {e}")
        return f"Recommended because it's similar to {target_product.get('Name', 'your selection')} and highly rated by customers."