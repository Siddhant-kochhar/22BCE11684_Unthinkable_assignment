"""
FastAPI Backend for E-commerce Recommendation System
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Union
from contextlib import asynccontextmanager
import os
import sys
import logging

# Add the parent directory to Python path to import ml module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ml.recommender import EcommerceRecommender
    from backend.gemini_helper import get_explanation, get_gemini_explainer
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all dependencies are installed and modules are in the correct location")


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global recommender instance
recommender = None

# Lifespan event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the recommendation system on startup"""
    global recommender
    try:
        data_path = "data/marketing_sample_for_walmart_com-walmart_com_product_review__20200701_20201231__5k_data.tsv"
        if not os.path.exists(data_path):
            # Try alternative path
            data_path = "marketing_sample_for_walmart_com-walmart_com_product_review__20200701_20201231__5k_data.tsv"
        
        if not os.path.exists(data_path):
            logger.warning(f"Data file not found at {data_path}. Server will start without recommendation system.")
            recommender = None
        else:
            logger.info(f"Loading data from: {data_path}")
            logger.info("This may take a moment...")
            recommender = EcommerceRecommender(data_path)
            logger.info("Recommendation system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize recommender: {e}")
        logger.info("Server will start in limited mode without recommendations")
        recommender = None
    
    yield
    
    # Cleanup on shutdown (if needed)
    logger.info("Shutting down recommendation system")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="E-commerce Recommendation System",
    description="AI-powered product recommendation system with Gemini explanations",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (for frontend - optional, serving HTML directly instead)
# app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Global recommender instance
recommender = None

# Pydantic models for request/response
class ProductResponse(BaseModel):
    ProdID: float
    Name: str
    Brand: str
    Rating: float
    ReviewCount: float
    ImageURL: str

class RecommendationResponse(BaseModel):
    ProdID: float
    Name: str
    Brand: str
    Rating: float
    ReviewCount: float
    ImageURL: str
    explanation: Optional[str] = None

class ErrorResponse(BaseModel):
    error: str
    message: str



@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main page"""
    try:
        with open("index.html", "r") as f:
            return HTMLResponse(content=f.read())
    except Exception as e:
        logger.error(f"Error serving index.html: {e}")
        return HTMLResponse(content=f"<h1>Error loading index.html: {e}</h1>")

@app.get("/main", response_class=HTMLResponse)
async def main_page():
    """Serve the main application page"""
    try:
        with open("main.html", "r") as f:
            return HTMLResponse(content=f.read())
    except Exception as e:
        logger.error(f"Error serving main.html: {e}")
        return HTMLResponse(content=f"<h1>Error loading main.html: {e}</h1>")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "E-commerce Recommendation System is running!"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "recommender_loaded": recommender is not None}

@app.get("/products", response_model=List[ProductResponse])
async def get_all_products(
    limit: int = Query(50, ge=1, le=500, description="Number of products to return"),
    offset: int = Query(0, ge=0, description="Number of products to skip")
):
    """Get all products with pagination"""
    try:
        if not recommender:
            raise HTTPException(status_code=500, detail="Recommender not initialized")
        
        all_products = recommender.get_all_products()
        
        # Apply pagination
        paginated_products = all_products[offset:offset + limit]
        
        return paginated_products
    except Exception as e:
        logger.error(f"Error fetching products: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/product/{product_id}")
async def get_product(product_id: Union[int, float]):
    """Get product details by ID"""
    try:
        if not recommender:
            raise HTTPException(status_code=500, detail="Recommender not initialized")
        
        product = recommender.get_product_by_id(product_id)
        if not product:
            raise HTTPException(status_code=404, detail=f"Product {product_id} not found")
        
        return product
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching product {product_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/trending", response_model=List[ProductResponse])
async def get_trending_products(
    limit: int = Query(10, ge=1, le=50, description="Number of trending products to return")
):
    """Get trending products"""
    try:
        if not recommender:
            raise HTTPException(status_code=500, detail="Recommender not initialized")
        
        trending = recommender.get_trending_products(top_n=limit)
        return trending
    except Exception as e:
        logger.error(f"Error fetching trending products: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommend/content/{product_id}", response_model=List[ProductResponse])
async def get_content_recommendations(
    product_id: Union[int, float],
    limit: int = Query(10, ge=1, le=20, description="Number of recommendations to return")
):
    """Get content-based recommendations for a product"""
    try:
        if not recommender:
            raise HTTPException(status_code=500, detail="Recommender not initialized")
        
        recommendations = recommender.content_based_recommendations(product_id, top_n=limit)
        
        if not recommendations:
            raise HTTPException(status_code=404, detail=f"No recommendations found for product {product_id}")
        
        return recommendations
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting content recommendations for {product_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommend/collaborative/{user_id}", response_model=List[ProductResponse])
async def get_collaborative_recommendations(
    user_id: Union[int, float],
    limit: int = Query(10, ge=1, le=20, description="Number of recommendations to return")
):
    """Get collaborative filtering recommendations for a user"""
    try:
        if not recommender:
            raise HTTPException(status_code=500, detail="Recommender not initialized")
        
        recommendations = recommender.collaborative_filtering_recommendations(user_id, top_n=limit)
        
        if not recommendations:
            raise HTTPException(status_code=404, detail=f"No recommendations found for user {user_id}")
        
        return recommendations
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting collaborative recommendations for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommend/hybrid/{product_id}/{user_id}", response_model=List[ProductResponse])
async def get_hybrid_recommendations(
    product_id: Union[int, float],
    user_id: Union[int, float],
    limit: int = Query(10, ge=1, le=20, description="Number of recommendations to return")
):
    """Get hybrid recommendations combining content-based and collaborative filtering"""
    try:
        if not recommender:
            raise HTTPException(status_code=500, detail="Recommender not initialized")
        
        recommendations = recommender.hybrid_recommendations(product_id, user_id, top_n=limit)
        
        if not recommendations:
            raise HTTPException(
                status_code=404, 
                detail=f"No recommendations found for product {product_id} and user {user_id}"
            )
        
        return recommendations
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting hybrid recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommend/content/{product_id}/explained", response_model=List[RecommendationResponse])
async def get_content_recommendations_with_explanation(
    product_id: Union[int, float],
    limit: int = Query(5, ge=1, le=10, description="Number of recommendations to return")
):
    """Get content-based recommendations with AI explanations"""
    try:
        if not recommender:
            raise HTTPException(status_code=500, detail="Recommender not initialized")
        
        # Get the target product
        target_product = recommender.get_product_by_id(product_id)
        if not target_product:
            raise HTTPException(status_code=404, detail=f"Product {product_id} not found")
        
        # Get recommendations
        recommendations = recommender.content_based_recommendations(product_id, top_n=limit)
        
        if not recommendations:
            raise HTTPException(status_code=404, detail=f"No recommendations found for product {product_id}")
        
        # Add explanations
        explained_recommendations = []
        for rec in recommendations:
            try:
                explanation = get_explanation(target_product, rec, "content-based")
                rec_with_explanation = RecommendationResponse(**rec, explanation=explanation)
                explained_recommendations.append(rec_with_explanation)
            except Exception as e:
                logger.warning(f"Could not generate explanation for product {rec.get('ProdID')}: {e}")
                # Add without explanation if Gemini fails
                rec_with_explanation = RecommendationResponse(
                    **rec, 
                    explanation=f"Recommended because it's similar to {target_product.get('Name')} in features and category."
                )
                explained_recommendations.append(rec_with_explanation)
        
        return explained_recommendations
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting explained recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommend/hybrid/{product_id}/{user_id}/explained", response_model=List[RecommendationResponse])
async def get_hybrid_recommendations_with_explanation(
    product_id: Union[int, float],
    user_id: Union[int, float],
    limit: int = Query(5, ge=1, le=10, description="Number of recommendations to return")
):
    """Get hybrid recommendations with AI explanations"""
    try:
        if not recommender:
            raise HTTPException(status_code=500, detail="Recommender not initialized")
        
        # Get the target product
        target_product = recommender.get_product_by_id(product_id)
        if not target_product:
            raise HTTPException(status_code=404, detail=f"Product {product_id} not found")
        
        # Get recommendations
        recommendations = recommender.hybrid_recommendations(product_id, user_id, top_n=limit)
        
        if not recommendations:
            raise HTTPException(
                status_code=404, 
                detail=f"No recommendations found for product {product_id} and user {user_id}"
            )
        
        # Add explanations
        explained_recommendations = []
        for rec in recommendations:
            try:
                explanation = get_explanation(target_product, rec, "hybrid")
                rec_with_explanation = RecommendationResponse(**rec, explanation=explanation)
                explained_recommendations.append(rec_with_explanation)
            except Exception as e:
                logger.warning(f"Could not generate explanation for product {rec.get('ProdID')}: {e}")
                # Add fallback explanation
                rec_with_explanation = RecommendationResponse(
                    **rec, 
                    explanation=f"Recommended based on similarity and customer preferences for {target_product.get('Name')}."
                )
                explained_recommendations.append(rec_with_explanation)
        
        return explained_recommendations
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting explained hybrid recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search")
async def search_products(
    query: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(20, ge=1, le=100, description="Number of results to return")
):
    """Search products by name or brand"""
    try:
        if not recommender:
            raise HTTPException(status_code=500, detail="Recommender not initialized")
        
        all_products = recommender.get_all_products()
        
        # Simple text search
        query_lower = query.lower()
        matching_products = [
            product for product in all_products
            if query_lower in product.get('Name', '').lower() or 
               query_lower in product.get('Brand', '').lower()
        ]
        
        return matching_products[:limit]
    except Exception as e:
        logger.error(f"Error searching products: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return {"error": "Internal server error", "message": str(exc)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)