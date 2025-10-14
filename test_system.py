"""
Test script for E-commerce Recommendation System
Run this to verify everything is working correctly
"""

import sys
import os
import asyncio

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def test_system():
    """Test the recommendation system components"""
    
    print("🧪 Testing E-commerce Recommendation System...")
    print("="*50)
    
    # Test 1: Import ML module
    print("\n1. Testing ML Module Import...")
    try:
        from ml.recommender import EcommerceRecommender
        print("✅ ML module imported successfully")
    except ImportError as e:
        print(f"❌ ML module import failed: {e}")
        return False
    
    # Test 2: Load data and initialize recommender
    print("\n2. Testing Data Loading...")
    try:
        data_path = "data/marketing_sample_for_walmart_com-walmart_com_product_review__20200701_20201231__5k_data.tsv"
        if not os.path.exists(data_path):
            data_path = "marketing_sample_for_walmart_com-walmart_com_product_review__20200701_20201231__5k_data.tsv"
        
        recommender = EcommerceRecommender(data_path)
        print(f"✅ Data loaded successfully. Shape: {recommender.data.shape}")
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False
    
    # Test 3: Get all products
    print("\n3. Testing Product Retrieval...")
    try:
        products = recommender.get_all_products()
        print(f"✅ Retrieved {len(products)} products")
        if products:
            print(f"   Sample product: {products[0]['Name']}")
    except Exception as e:
        print(f"❌ Product retrieval failed: {e}")
        return False
    
    # Test 4: Get trending products
    print("\n4. Testing Trending Products...")
    try:
        trending = recommender.get_trending_products(top_n=5)
        print(f"✅ Retrieved {len(trending)} trending products")
        if trending:
            print(f"   Top trending: {trending[0]['Name']}")
    except Exception as e:
        print(f"❌ Trending products failed: {e}")
        return False
    
    # Test 5: Content-based recommendations
    print("\n5. Testing Content-Based Recommendations...")
    try:
        if products:
            product_id = products[0]['ProdID']
            recs = recommender.content_based_recommendations(product_id, top_n=3)
            print(f"✅ Generated {len(recs)} content-based recommendations")
            if recs:
                print(f"   First recommendation: {recs[0]['Name']}")
    except Exception as e:
        print(f"❌ Content-based recommendations failed: {e}")
        return False
    
    # Test 6: Collaborative filtering recommendations
    print("\n6. Testing Collaborative Filtering...")
    try:
        user_id = 4  # Sample user ID
        collab_recs = recommender.collaborative_filtering_recommendations(user_id, top_n=3)
        print(f"✅ Generated {len(collab_recs)} collaborative filtering recommendations")
        if collab_recs:
            print(f"   First recommendation: {collab_recs[0]['Name']}")
    except Exception as e:
        print(f"❌ Collaborative filtering failed: {e}")
        print("   This might be normal if user ID doesn't exist in dataset")
    
    # Test 7: Hybrid recommendations
    print("\n7. Testing Hybrid Recommendations...")
    try:
        if products:
            product_id = products[0]['ProdID']
            user_id = 4
            hybrid_recs = recommender.hybrid_recommendations(product_id, user_id, top_n=3)
            print(f"✅ Generated {len(hybrid_recs)} hybrid recommendations")
            if hybrid_recs:
                print(f"   First recommendation: {hybrid_recs[0]['Name']}")
    except Exception as e:
        print(f"❌ Hybrid recommendations failed: {e}")
    
    # Test 8: Gemini helper (if API key is available)
    print("\n8. Testing Gemini AI Helper...")
    try:
        from backend.gemini_helper import GeminiExplainer
        
        # Check if API key is available
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key and api_key != "your_gemini_api_key_here":
            explainer = GeminiExplainer(api_key)
            
            # Test with sample products
            if products and len(products) >= 2:
                target_product = products[0]
                recommended_product = products[1]
                
                explanation = explainer.get_recommendation_explanation(
                    target_product, recommended_product, "content-based"
                )
                print("✅ Gemini AI explanation generated successfully")
                print(f"   Sample explanation: {explanation[:100]}...")
            else:
                print("✅ Gemini helper initialized (no test data)")
        else:
            print("⚠️ Gemini API key not found. Set GEMINI_API_KEY environment variable to test AI features.")
    except Exception as e:
        print(f"❌ Gemini helper failed: {e}")
        print("   Make sure google-generativeai is installed and API key is set")
    
    print("\n" + "="*50)
    print("🎉 System test completed!")
    print("\nNext steps:")
    print("1. Set GEMINI_API_KEY in your environment")
    print("2. Run: python backend/main.py")
    print("3. Open: http://localhost:8000")
    print("4. API docs: http://localhost:8000/docs")
    
    return True

if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    asyncio.run(test_system())