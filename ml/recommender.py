"""
E-commerce Recommendation System ML Module
Contains all recommendation algorithms: Content-based, Collaborative Filtering, Hybrid
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import os
from typing import Dict, List, Optional, Union


class EcommerceRecommender:
    """
    Comprehensive recommendation system for e-commerce products
    Supports content-based, collaborative filtering, and hybrid recommendations
    """
    
    def __init__(self, data_path: str):
        """
        Initialize the recommender with product data
        
        Args:
            data_path: Path to the TSV data file
        """
        self.data_path = data_path
        self.data = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.similarity_matrix = None
        self.user_item_matrix = None
        self.nlp = None
        self._models_initialized = False
        
        # Load data immediately but prepare models lazily
        self._load_and_preprocess_data()
        print("✅ Recommender initialized - models will load on first use")
    
    def _ensure_models_ready(self):
        """Initialize ML models if not already done"""
        if not self._models_initialized:
            print("🔄 Initializing ML models...")
            self._prepare_content_based_model()
            self._models_initialized = True
            print("✅ ML models ready!")
        
    def _load_and_preprocess_data(self):
        """Load and preprocess the dataset"""
        try:
            # Load data
            self.data = pd.read_csv(self.data_path, sep='\t')
            
            # Select relevant columns
            columns_to_keep = [
                'Uniq Id', 'Product Id', 'Product Rating', 'Product Reviews Count',
                'Product Category', 'Product Brand', 'Product Name', 'Product Image Url',
                'Product Description', 'Product Tags'
            ]
            self.data = self.data[columns_to_keep]
            
            # Rename columns for easier handling
            column_mapping = {
                'Uniq Id': 'ID',
                'Product Id': 'ProdID',
                'Product Rating': 'Rating',
                'Product Reviews Count': 'ReviewCount',
                'Product Category': 'Category',
                'Product Brand': 'Brand',
                'Product Name': 'Name',
                'Product Image Url': 'ImageURL',
                'Product Description': 'Description',
                'Product Tags': 'Tags'
            }
            self.data.rename(columns=column_mapping, inplace=True)
            
            # Handle missing values (pandas 3.0 compatible)
            fill_values = {
                'Rating': 0,
                'ReviewCount': 0,
                'Category': '',
                'Brand': '',
                'Description': '',
                'Tags': ''
            }
            self.data = self.data.fillna(value=fill_values)
            
            # Extract numeric IDs (pandas 3.0 compatible)
            id_numeric = self.data['ID'].str.extract(r'(\d+)').astype(float)
            prodid_numeric = self.data['ProdID'].str.extract(r'(\d+)').astype(float)
            self.data = self.data.assign(ID=id_numeric, ProdID=prodid_numeric)
            
            # Clean and create tags
            self._create_enhanced_tags()
            
            print(f"Data loaded successfully. Shape: {self.data.shape}")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def _create_enhanced_tags(self):
        """Create enhanced tags using NLP processing"""
        try:
            # Load spaCy model
            self.nlp = spacy.load("en_core_web_sm")
            
            def clean_and_extract_tags(text):
                """Extract meaningful tags from text"""
                if pd.isna(text) or text == '':
                    return ''
                doc = self.nlp(str(text).lower())
                tags = [token.text for token in doc if token.text.isalnum() and token.text not in STOP_WORDS]
                return ', '.join(tags)
            
            # Apply tag extraction to relevant columns (pandas 3.0 compatible)
            columns_for_tags = ['Category', 'Brand', 'Description']
            for column in columns_for_tags:
                if column in self.data.columns:
                    cleaned_column = self.data[column].apply(clean_and_extract_tags)
                    self.data = self.data.assign(**{column: cleaned_column})
            
            # Combine all tags
            combined_tags = self.data[columns_for_tags].apply(
                lambda row: ', '.join([str(val) for val in row if str(val) != '']), axis=1
            )
            self.data = self.data.assign(Tags=combined_tags)
            
        except Exception as e:
            print(f"Warning: Could not load spaCy model. Using basic tag processing. Error: {e}")
            # Fallback to basic tag processing (pandas 3.0 compatible)
            basic_tags = self.data[['Category', 'Brand', 'Description']].apply(
                lambda row: ', '.join([str(val) for val in row if pd.notna(val) and str(val) != '']), axis=1
            )
            self.data = self.data.assign(Tags=basic_tags)
    
    def _prepare_content_based_model(self):
        """Prepare TF-IDF vectors for content-based recommendations"""
        try:
            self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.data['Tags'])
            self.similarity_matrix = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
            print("Content-based model prepared successfully")
        except Exception as e:
            print(f"Error preparing content-based model: {e}")
            raise
    
    def get_all_products(self) -> List[Dict]:
        """Get all products in the dataset"""
        return self.data[['ProdID', 'Name', 'Brand', 'Rating', 'ReviewCount', 'ImageURL']].to_dict('records')
    
    def get_product_by_id(self, product_id: Union[int, float]) -> Optional[Dict]:
        """Get product details by ID"""
        try:
            product = self.data[self.data['ProdID'] == float(product_id)]
            if not product.empty:
                return product.iloc[0].to_dict()
            return None
        except Exception as e:
            print(f"Error fetching product {product_id}: {e}")
            return None
    
    def get_product_by_name(self, product_name: str) -> Optional[Dict]:
        """Get product details by name"""
        try:
            product = self.data[self.data['Name'] == product_name]
            if not product.empty:
                return product.iloc[0].to_dict()
            return None
        except Exception as e:
            print(f"Error fetching product {product_name}: {e}")
            return None
    
    def content_based_recommendations(self, product_identifier: Union[int, str], top_n: int = 10) -> List[Dict]:
        """
        Get content-based recommendations for a product
        
        Args:
            product_identifier: Product ID (int) or Product Name (str)
            top_n: Number of recommendations to return
            
        Returns:
            List of recommended products
        """
        self._ensure_models_ready()
        try:
            # Find the product
            if isinstance(product_identifier, (int, float)):
                product_data = self.data[self.data['ProdID'] == float(product_identifier)]
            else:
                product_data = self.data[self.data['Name'] == product_identifier]
            
            if product_data.empty:
                print(f"Product '{product_identifier}' not found")
                return []
            
            item_index = product_data.index[0]
            
            # Get similarity scores
            similar_items = list(enumerate(self.similarity_matrix[item_index]))
            similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)
            
            # Get top N similar items (excluding the item itself)
            top_similar_items = similar_items[1:top_n+1]
            recommended_indices = [x[0] for x in top_similar_items]
            
            # Return product details
            recommendations = self.data.iloc[recommended_indices][
                ['ProdID', 'Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']
            ].to_dict('records')
            
            return recommendations
            
        except Exception as e:
            print(f"Error in content-based recommendations: {e}")
            return []
    
    def collaborative_filtering_recommendations(self, user_id: Union[int, float], top_n: int = 10) -> List[Dict]:
        """
        Get collaborative filtering recommendations for a user
        
        Args:
            user_id: User ID
            top_n: Number of recommendations to return
            
        Returns:
            List of recommended products
        """
        self._ensure_models_ready()
        try:
            # Create user-item matrix
            user_item_matrix = self.data.pivot_table(
                index='ID', columns='ProdID', values='Rating', aggfunc='mean'
            ).fillna(0)
            
            if user_id not in user_item_matrix.index:
                print(f"User {user_id} not found")
                return []
            
            # Calculate user similarity
            user_similarity = cosine_similarity(user_item_matrix)
            target_user_index = user_item_matrix.index.get_loc(user_id)
            
            # Get similar users
            user_similarities = user_similarity[target_user_index]
            similar_users_indices = user_similarities.argsort()[::-1][1:]
            
            # Generate recommendations
            recommended_items = []
            for user_index in similar_users_indices:
                rated_by_similar_user = user_item_matrix.iloc[user_index]
                not_rated_by_target_user = (
                    (rated_by_similar_user > 0) & 
                    (user_item_matrix.iloc[target_user_index] == 0)
                )
                
                recommended_items.extend(
                    user_item_matrix.columns[not_rated_by_target_user][:top_n]
                )
                
                if len(recommended_items) >= top_n:
                    break
            
            # Get product details
            recommended_items = list(set(recommended_items))[:top_n]
            recommendations = self.data[self.data['ProdID'].isin(recommended_items)][
                ['ProdID', 'Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']
            ].to_dict('records')
            
            return recommendations[:top_n]
            
        except Exception as e:
            print(f"Error in collaborative filtering: {e}")
            return []
    
    def hybrid_recommendations(self, product_identifier: Union[int, str], user_id: Union[int, float], top_n: int = 10) -> List[Dict]:
        """
        Get hybrid recommendations combining content-based and collaborative filtering
        
        Args:
            product_identifier: Product ID or Name
            user_id: User ID
            top_n: Number of recommendations to return
            
        Returns:
            List of recommended products
        """
        self._ensure_models_ready()
        try:
            # Get both types of recommendations
            content_recs = self.content_based_recommendations(product_identifier, top_n)
            collab_recs = self.collaborative_filtering_recommendations(user_id, top_n)
            
            # Combine and deduplicate
            all_recs = content_recs + collab_recs
            seen_products = set()
            hybrid_recs = []
            
            for rec in all_recs:
                if rec['ProdID'] not in seen_products:
                    hybrid_recs.append(rec)
                    seen_products.add(rec['ProdID'])
                
                if len(hybrid_recs) >= top_n:
                    break
            
            return hybrid_recs
            
        except Exception as e:
            print(f"Error in hybrid recommendations: {e}")
            return []
    
    def get_trending_products(self, top_n: int = 10) -> List[Dict]:
        """Get trending products based on ratings and review counts"""
        try:
            # Calculate weighted rating (rating * log(review_count + 1)) - pandas 3.0 compatible
            weighted_score = self.data['Rating'] * np.log(self.data['ReviewCount'] + 1)
            data_with_score = self.data.assign(weighted_score=weighted_score)
            
            trending = data_with_score.nlargest(top_n, 'weighted_score')[
                ['ProdID', 'Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']
            ].to_dict('records')
            
            return trending
            
        except Exception as e:
            print(f"Error getting trending products: {e}")
            return []