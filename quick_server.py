#!/usr/bin/env python3
"""
Quick FastAPI server for testing - minimal data loading
"""

import os
import sys
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from typing import List
from pydantic import BaseModel
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple product model
class Product(BaseModel):
    product_id: str
    product_name: str
    brand: str
    price: float = 0.0

# Initialize FastAPI app
app = FastAPI(
    title="Quick E-commerce Server",
    description="Minimal server for testing",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Try to mount frontend if it exists
try:
    if os.path.exists("frontend"):
        app.mount("/static", StaticFiles(directory="frontend"), name="static")
except Exception as e:
    logger.warning(f"Could not mount frontend: {e}")

# Global data
sample_products = []

# Load minimal data on startup
try:
    data_path = "marketing_sample_for_walmart_com-walmart_com_product_review__20200701_20201231__5k_data.tsv"
    if os.path.exists(data_path):
        logger.info("Loading sample data...")
        df = pd.read_csv(data_path, sep='\t', nrows=100)  # Load only first 100 rows
        
        # Create sample products
        for _, row in df.head(20).iterrows():  # Use only first 20
            try:
                product = Product(
                    product_id=str(row.get('Product Id', 'unknown')),
                    product_name=str(row.get('Product Name', 'Unknown Product')),
                    brand=str(row.get('Product Brand', 'Unknown Brand')),
                    price=float(str(row.get('Product Price', '0')).replace('$', '').replace(',', '') or 0)
                )
                sample_products.append(product)
            except Exception as e:
                logger.warning(f"Error processing row: {e}")
                continue
        
        logger.info(f"Loaded {len(sample_products)} sample products")
    else:
        logger.warning("Data file not found, using mock data")
        # Create mock data
        for i in range(10):
            sample_products.append(Product(
                product_id=f"prod_{i}",
                product_name=f"Sample Product {i}",
                brand=f"Brand {i % 3}",
                price=float(10 + i * 5)
            ))
except Exception as e:
    logger.error(f"Error loading data: {e}")
    # Create mock data as fallback
    for i in range(5):
        sample_products.append(Product(
            product_id=f"mock_{i}",
            product_name=f"Mock Product {i}",
            brand="Mock Brand",
            price=float(10 + i * 2)
        ))

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main page"""
    try:
        if os.path.exists("frontend/index.html"):
            with open("frontend/index.html", "r") as f:
                return HTMLResponse(content=f.read())
    except Exception as e:
        logger.error(f"Error serving frontend: {e}")
    
    return HTMLResponse(content="""
    <html>
        <body>
            <h1>Quick E-commerce Server</h1>
            <p>API is running! Try these endpoints:</p>
            <ul>
                <li><a href="/products">/products</a> - Get products</li>
                <li><a href="/health">/health</a> - Health check</li>
                <li><a href="/docs">/docs</a> - API documentation</li>
            </ul>
        </body>
    </html>
    """)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "products_loaded": len(sample_products),
        "message": "Quick server is running!"
    }

@app.get("/products", response_model=List[Product])
async def get_products(
    limit: int = Query(default=10, ge=1, le=50),
    offset: int = Query(default=0, ge=0)
):
    """Get products with pagination"""
    try:
        total = len(sample_products)
        start = min(offset, total)
        end = min(offset + limit, total)
        
        products = sample_products[start:end]
        logger.info(f"Returning {len(products)} products (offset={offset}, limit={limit})")
        return products
    except Exception as e:
        logger.error(f"Error getting products: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/search")
async def search_products(q: str = Query(..., description="Search query")):
    """Search products"""
    try:
        if not q:
            return sample_products[:10]
        
        # Simple search
        q_lower = q.lower()
        results = []
        for product in sample_products:
            if (q_lower in product.product_name.lower() or 
                q_lower in product.brand.lower()):
                results.append(product)
        
        return results[:10]
    except Exception as e:
        logger.error(f"Error searching products: {e}")
        raise HTTPException(status_code=500, detail="Search error")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting quick server...")
    logger.info(f"Loaded {len(sample_products)} products")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")