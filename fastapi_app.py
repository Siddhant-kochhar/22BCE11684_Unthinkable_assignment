from fastapi import FastAPI, Request, Form, HTTPException, Depends, Cookie
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from contextlib import asynccontextmanager
import logging
import gspread
from google.auth import default
import json
import os
import hashlib
from datetime import datetime, timedelta
import uuid
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LLM Integration
try:
    import google.generativeai as genai
    # Configure Gemini using existing API key
    api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY', 'demo_key')
    genai.configure(api_key=api_key)
    LLM_AVAILABLE = True if api_key and api_key != 'demo_key' else False
    logger.info(f"✅ LLM Available: {LLM_AVAILABLE}")
except ImportError:
    LLM_AVAILABLE = False
    logger.warning("Google GenerativeAI not available. Install with: pip install google-generativeai")

# Global variables for data
trending_products = None
train_data = None
users_sheet = None
user_clicks = {}  # Store user click history: {user_id: [clicked_product_names]}
user_sessions = {}  # Store active sessions: {session_id: user_id}
logged_in_users = {}  # Store currently logged in users: {user_id: session_info}

# Google Sheets Authentication
class GoogleSheetsAuth:
    def __init__(self):
        self.gc = None
        self.sheet = None
        
    async def initialize(self):
        """Initialize Google Sheets connection"""
        try:
            # Load existing users from file if it exists
            import json
            import os
            
            users_file = "users_db.json"
            if os.path.exists(users_file):
                with open(users_file, "r") as f:
                    self.users_db = json.load(f)
                logger.info(f"✅ Loaded {len(self.users_db)} users from file")
            else:
                # Initialize with empty user database (no default users)
                self.users_db = {}
                self.save_users()
                logger.info("✅ Initialized empty user database")
            
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize authentication: {e}")
            # Fallback to empty in-memory user store
            self.users_db = {}
            return False
    
    def save_users(self):
        """Save users to file for persistence"""
        try:
            import json
            with open("users_db.json", "w") as f:
                json.dump(self.users_db, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save users: {e}")
    
    def authenticate_user(self, username: str, password: str) -> bool:
        """Check if user credentials are valid"""
        return self.users_db.get(username) == password
    
    def register_user(self, username: str, email: str, password: str) -> bool:
        """Register a new user"""
        if username not in self.users_db:
            self.users_db[username] = password
            self.save_users()  # Persist to file
            logger.info(f"New user registered: {username}")
            return True
        return False

# Global auth instance
auth_system = GoogleSheetsAuth()

# Session management functions
def create_session(user_id: str) -> str:
    """Create a new session for user"""
    session_id = str(uuid.uuid4())
    user_sessions[session_id] = user_id
    logged_in_users[user_id] = {
        "session_id": session_id,
        "login_time": datetime.now(),
        "last_activity": datetime.now()
    }
    logger.info(f"✅ Session created for user: {user_id}")
    return session_id

def get_user_from_session(session_id: str) -> str:
    """Get user ID from session"""
    return user_sessions.get(session_id, None)

def is_user_logged_in(session_id: str) -> bool:
    """Check if session is valid"""
    if not session_id:
        return False
    user_id = user_sessions.get(session_id)
    if user_id and user_id in logged_in_users:
        # Update last activity
        logged_in_users[user_id]["last_activity"] = datetime.now()
        return True
    return False

def logout_user(session_id: str):
    """Logout user and clear session"""
    if session_id in user_sessions:
        user_id = user_sessions[session_id]
        del user_sessions[session_id]
        if user_id in logged_in_users:
            del logged_in_users[user_id]
        logger.info(f"👋 User logged out: {user_id}")

# Dependency to get current user
def get_current_user(session_id: str = Cookie(default=None)):
    """Get current logged in user"""
    if session_id and is_user_logged_in(session_id):
        return get_user_from_session(session_id)
    return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load data on startup"""
    global trending_products, train_data
    try:
        logger.info("Loading data files...")
        
        # Initialize authentication system
        await auth_system.initialize()
        
        # Load data files
        trending_products = pd.read_csv("trending_products.csv")
        train_data = pd.read_csv("clean_data.csv")
        
        logger.info(f"✅ Loaded {len(trending_products)} trending products")
        logger.info(f"✅ Loaded {len(train_data)} training records")
        
    except Exception as e:
        logger.error(f"❌ Error loading data: {e}")
        # Create some sample data as fallback
        trending_products = pd.DataFrame({
            'Name': ['Sample Product 1', 'Sample Product 2', 'Sample Product 3'],
            'Brand': ['Brand A', 'Brand B', 'Brand C'],
            'ReviewCount': [100, 200, 150],
            'Rating': [4.5, 4.0, 4.7],
            'ImageURL': ['img_1.png', 'img_2.png', 'img_3.png']
        })
        train_data = trending_products.copy()
        train_data['Tags'] = ['electronics gadget', 'clothing fashion', 'book education']
        logger.info("Using fallback sample data")
    
    yield
    
    logger.info("Shutting down...")

# Initialize FastAPI app
app = FastAPI(
    title="E-commerce Recommendation System",
    description="FastAPI version with ML recommendations",
    version="1.0.0",
    lifespan=lifespan
)

# Setup templates
templates = Jinja2Templates(directory=".")

# Mount static files for images
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
    # Also serve images directly from root for compatibility
    app.mount("/img", StaticFiles(directory="static"), name="images")
except Exception as e:
    logger.warning(f"Could not mount static files: {e}")

# Utility functions
def truncate(text, length):
    """Function to truncate product name"""
    if len(text) > length:
        return text[:length] + "..."
    else:
        return text

def get_smart_image(product_name, brand=None):
    """Get appropriate image based on product type"""
    product_lower = str(product_name).lower()
    
    # Beauty & Cosmetics
    if any(word in product_lower for word in ['makeup', 'lipstick', 'nail', 'polish', 'mascara', 'foundation', 'blush', 'eyeshadow']):
        return random.choice(beauty_images)
    
    # Hair & Skincare
    elif any(word in product_lower for word in ['shampoo', 'conditioner', 'hair', 'cream', 'lotion', 'serum', 'moisturizer']):
        return random.choice(beauty_images)
    
    # Health & Personal Care
    elif any(word in product_lower for word in ['toothpaste', 'soap', 'razor', 'deodorant', 'vitamins', 'supplements']):
        return random.choice(health_images)
    
    # Default
    else:
        return random.choice(lifestyle_images)

async def generate_recommendation_explanation(recommended_product, user_behavior, similarity_score):
    """Generate LLM-powered explanation for why a product is recommended"""
    try:
        if not LLM_AVAILABLE:
            return f"Recommended based on {similarity_score:.1%} similarity to your interests."
        
        # Extract user interests from behavior
        clicked_products = [click['product_name'] for click in user_behavior[-5:]]  # Last 5 clicks
        
        # Create prompt for LLM
        prompt = f"""
        You are a helpful e-commerce recommendation assistant. Explain in 1-2 sentences why "{recommended_product['Name']}" is recommended to this user.

        User's Recent Activity:
        - Recently viewed: {', '.join(clicked_products)}
        
        Recommended Product:
        - Name: {recommended_product['Name']}
        - Brand: {recommended_product['Brand']}
        - Category: {recommended_product['Category']}
        - Similarity Score: {similarity_score:.1%}
        
        Generate a friendly, personalized explanation focusing on:
        1. Connection to user's interests
        2. Product benefits
        3. Why it's a good match
        
        Keep it conversational and under 40 words.
        """
        
        model = genai.GenerativeModel('gemini-2.5-flash')  # Updated model name
        response = model.generate_content(prompt)
        
        return response.text.strip()
        
    except Exception as e:
        logger.error(f"LLM explanation error: {e}")
        return f"Great match for your interests! {similarity_score:.1%} similarity to products you've viewed."

def content_based_recommendations(train_data, item_name, top_n=10):
    """Get content-based recommendations"""
    try:
        # Check if the item name exists in the training data
        if item_name not in train_data['Name'].values:
            print(f"Item '{item_name}' not found in the training data.")
            return pd.DataFrame()

        # Create a TF-IDF vectorizer for item descriptions
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')

        # Apply TF-IDF vectorization to item descriptions
        tfidf_matrix_content = tfidf_vectorizer.fit_transform(train_data['Tags'])

        # Calculate cosine similarity between items based on descriptions
        cosine_similarities_content = cosine_similarity(tfidf_matrix_content, tfidf_matrix_content)

        # Find the index of the item
        item_index = train_data[train_data['Name'] == item_name].index[0]

        # Get the cosine similarity scores for the item
        similar_items = list(enumerate(cosine_similarities_content[item_index]))

        # Sort similar items by similarity score in descending order
        similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)

        # Get the top N most similar items (excluding the item itself)
        top_similar_items = similar_items[1:top_n+1]

        # Get the indices of the top similar items
        recommended_item_indices = [x[0] for x in top_similar_items]

        # Get the details of the top similar items
        recommended_items_details = train_data.iloc[recommended_item_indices][['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']].copy()
        
        # Add similarity scores to the dataframe
        similarity_scores = [x[1] for x in top_similar_items]
        recommended_items_details['similarity'] = similarity_scores
        
        # Add Category column if it exists
        if 'Category' in train_data.columns:
            recommended_items_details['Category'] = train_data.iloc[recommended_item_indices]['Category'].values

        return recommended_items_details
    except Exception as e:
        logger.error(f"Error in recommendations: {e}")
        return pd.DataFrame()

# Enhanced Image URLs with categories
beauty_images = [
    "/static/img_1.png",  # Beauty/cosmetics
    "/static/img_2.png",  # Skincare
    "/static/img_3.png",  # Makeup
]

health_images = [
    "/static/img_4.png",  # Health products
    "/static/img_5.png",  # Supplements
    "/static/img_6.png",  # Personal care
]

lifestyle_images = [
    "/static/img_7.png",  # Lifestyle products
    "/static/img_8.png",  # General items
]

# Combined image pool
random_image_urls = beauty_images + health_images + lifestyle_images

price_list = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]

# Routes
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main page with trending products"""
    try:
        if trending_products is None:
            raise HTTPException(status_code=500, detail="Data not loaded")
        
        # Create smart image URLs for each product based on product type
        products_to_show = trending_products.head(8)
        smart_product_image_urls = []
        
        for _, product in products_to_show.iterrows():
            smart_image = get_smart_image(product['Name'], product.get('Brand', ''))
            smart_product_image_urls.append(smart_image)
        
        context = {
            "request": request,
            "trending_products": products_to_show,
            "truncate": truncate,
            "random_product_image_urls": smart_product_image_urls,
            "random_price": random.choice(price_list),
            "get_smart_image": get_smart_image  # Add the function for templates
        }
        
        return templates.TemplateResponse("index.html", context)
    except Exception as e:
        logger.error(f"Error in index route: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading page: {e}")

@app.get("/main", response_class=HTMLResponse)
async def main(request: Request):
    """Main application page"""
    try:
        context = {
            "request": request,
            "content_based_rec": pd.DataFrame(),  # Empty DataFrame for initial load
            "message": None,
            "truncate": truncate,
            "random_product_image_urls": [],
            "random_price": random.choice(price_list)
        }
        return templates.TemplateResponse("main.html", context)
    except Exception as e:
        logger.error(f"Error in main route: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading main page: {e}")

@app.get("/index", response_class=HTMLResponse)
async def index_redirect(request: Request):
    """Redirect to index page"""
    return await index(request)

@app.post("/signup", response_class=HTMLResponse)
async def signup(
    request: Request,
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...)
):
    """Handle user signup with automatic login"""
    try:
        logger.info(f"New signup attempt: {username} ({email})")
        
        # Register user with Google Sheets auth system
        if auth_system.register_user(username, email, password):
            # Auto-login after successful signup
            session_id = create_session(username)
            
            message = f"Welcome {username}! You have been registered and logged in successfully!"
            logger.info(f"✅ User {username} registered and logged in successfully")
            
            products_to_show = trending_products.head(8)
            smart_product_image_urls = []
            
            for _, product in products_to_show.iterrows():
                smart_image = get_smart_image(product['Name'], product.get('Brand', ''))
                smart_product_image_urls.append(smart_image)
            
            context = {
                "request": request,
                "trending_products": products_to_show,
                "truncate": truncate,
                "random_product_image_urls": smart_product_image_urls,
                "random_price": random.choice(price_list),
                "signup_message": message,
                "current_user": username,
                "get_smart_image": get_smart_image
            }
            
            # Create response with session cookie
            response = templates.TemplateResponse("index.html", context)
            response.set_cookie(
                key="session_id", 
                value=session_id, 
                max_age=86400,  # 24 hours
                httponly=True,
                secure=False  # Set to True in production with HTTPS
            )
            return response
            
        else:
            message = f"Username {username} already exists. Please try a different username."
            logger.warning(f"⚠️ Registration failed - username {username} exists")
            
            products_to_show = trending_products.head(8)
            smart_product_image_urls = []
            
            for _, product in products_to_show.iterrows():
                smart_image = get_smart_image(product['Name'], product.get('Brand', ''))
                smart_product_image_urls.append(smart_image)
            
            context = {
                "request": request,
                "trending_products": products_to_show,
                "truncate": truncate,
                "random_product_image_urls": smart_product_image_urls,
                "random_price": random.choice(price_list),
                "signup_message": message,
                "current_user": None,
                "get_smart_image": get_smart_image
            }
            
            return templates.TemplateResponse("index.html", context)
            
    except Exception as e:
        logger.error(f"Error in signup: {e}")
        raise HTTPException(status_code=500, detail="Signup error")

@app.get("/gform-redirect", response_class=HTMLResponse)
async def gform_redirect(
    request: Request,
    username: str = None,
    email: str = None
):
    """Handle users returning from Google Form"""
    logger.info(f"User returning from Google Form: {username} ({email})")
    
    products_to_show = trending_products.head(8)
    smart_product_image_urls = []
    
    for _, product in products_to_show.iterrows():
        smart_image = get_smart_image(product['Name'], product.get('Brand', ''))
        smart_product_image_urls.append(smart_image)
    
    context = {
        "request": request,
        "trending_products": products_to_show,
        "truncate": truncate,
        "random_product_image_urls": smart_product_image_urls,
        "random_price": random.choice(price_list),
        "gform_data": {"username": username, "email": email} if username or email else None,
        "current_user": None,
        "get_smart_image": get_smart_image
    }
    
    return templates.TemplateResponse("index.html", context)

@app.get("/debug-users")
async def debug_users():
    """Debug endpoint to see registered users"""
    return {
        "users_count": len(auth_system.users_db),
        "usernames": list(auth_system.users_db.keys()),
        "sessions_count": len(user_sessions),
        "logged_in_count": len(logged_in_users)
    }

@app.get("/debug-clicks")
async def debug_clicks(current_user: str = Depends(get_current_user)):
    """Debug endpoint to see user clicks"""
    if not current_user:
        return {"error": "Not logged in"}
    
    return {
        "user": current_user,
        "clicks": user_clicks.get(current_user, []),
        "total_clicks": len(user_clicks.get(current_user, []))
    }

@app.get("/llm-demo")
async def llm_demo():
    """Demo endpoint to show LLM capabilities"""
    return {
        "llm_available": LLM_AVAILABLE,
        "description": "This endpoint shows if LLM explanations are available",
        "setup_instructions": "Set GOOGLE_API_KEY in .env file to enable LLM explanations",
        "fallback": "System works without LLM - provides basic explanations"
    }

@app.post("/sync-gform")
async def sync_google_form():
    """Endpoint to sync Google Form responses with our user database"""
    try:
        # This would connect to Google Sheets API to fetch form responses
        # For demonstration, we'll return a success message
        logger.info("Google Form sync requested")
        return {"status": "success", "message": "Google Form sync completed", "synced_users": 0}
    except Exception as e:
        logger.error(f"Google Form sync error: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/signin", response_class=HTMLResponse)
async def signin(
    request: Request,
    signinUsername: str = Form(...),
    signinPassword: str = Form(...)
):
    """Handle user signin with session management"""
    try:
        logger.info(f"Signin attempt: {signinUsername}")
        
        # Authenticate user with Google Sheets auth system
        if auth_system.authenticate_user(signinUsername, signinPassword):
            # Create session for authenticated user
            session_id = create_session(signinUsername)
            
            message = f"Welcome back {signinUsername}! You have been signed in successfully!"
            logger.info(f"✅ User {signinUsername} authenticated successfully")
            
            products_to_show = trending_products.head(8)
            smart_product_image_urls = []
            
            for _, product in products_to_show.iterrows():
                smart_image = get_smart_image(product['Name'], product.get('Brand', ''))
                smart_product_image_urls.append(smart_image)
            
            context = {
                "request": request,
                "trending_products": products_to_show,
                "truncate": truncate,
                "random_product_image_urls": smart_product_image_urls,
                "random_price": random.choice(price_list),
                "signup_message": message,
                "current_user": signinUsername,
                "get_smart_image": get_smart_image
            }
            
            # Create response with session cookie
            response = templates.TemplateResponse("index.html", context)
            response.set_cookie(
                key="session_id", 
                value=session_id, 
                max_age=86400,  # 24 hours
                httponly=True,
                secure=False  # Set to True in production with HTTPS
            )
            return response
            
        else:
            message = "Invalid username or password. Please try again."
            logger.warning(f"⚠️ Authentication failed for {signinUsername}")
            
            products_to_show = trending_products.head(8)
            smart_product_image_urls = []
            
            for _, product in products_to_show.iterrows():
                smart_image = get_smart_image(product['Name'], product.get('Brand', ''))
                smart_product_image_urls.append(smart_image)
            
            context = {
                "request": request,
                "trending_products": products_to_show,
                "truncate": truncate,
                "random_product_image_urls": smart_product_image_urls,
                "random_price": random.choice(price_list),
                "signup_message": message,
                "current_user": None,
                "get_smart_image": get_smart_image
            }
            
            return templates.TemplateResponse("index.html", context)
            
    except Exception as e:
        logger.error(f"Error in signin: {e}")
        raise HTTPException(status_code=500, detail="Signin error")

@app.get("/recommendations", response_class=HTMLResponse)
async def get_recommendations(request: Request):
    """GET recommendations page"""
    return await main(request)

@app.post("/recommendations", response_class=HTMLResponse)
async def post_recommendations(
    request: Request,
    prod: str = Form(default=""),
    nbr: str = Form(default="10")
):
    """Get product recommendations"""
    try:
        # Handle empty values like the original Flask code
        if not prod or prod.strip() == "":
            context = {
                "request": request,
                "content_based_rec": pd.DataFrame(),
                "message": "Please enter a product name.",
                "truncate": truncate,
                "random_product_image_urls": [],
                "random_price": random.choice(price_list)
            }
            return templates.TemplateResponse("main.html", context)
        
        # Convert nbr to int, default to 10 if invalid
        try:
            nbr_int = int(nbr) if nbr and nbr.strip() else 10
        except ValueError:
            nbr_int = 10
        
        logger.info(f"Getting recommendations for: {prod} (top {nbr_int})")
        
        if train_data is None:
            raise HTTPException(status_code=500, detail="Training data not available")
        
        content_based_rec = content_based_recommendations(train_data, prod, top_n=nbr_int)
        
        if content_based_rec.empty:
            context = {
                "request": request,
                "content_based_rec": pd.DataFrame(),
                "message": "No recommendations available for this product.",
                "truncate": truncate,
                "random_product_image_urls": [],
                "random_price": random.choice(price_list)
            }
            return templates.TemplateResponse("main.html", context)
        else:
            # Create a list of random image URLs for each recommended product
            # Use trending_products length like in original code
            random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
            
            logger.info(f"Found {len(content_based_rec)} recommendations")
            logger.info(f"Content based rec: {content_based_rec}")
            logger.info(f"Random image URLs: {random_product_image_urls}")
            
            context = {
                "request": request,
                "content_based_rec": content_based_rec,
                "truncate": truncate,
                "random_product_image_urls": random_product_image_urls,
                "random_price": random.choice(price_list),
                "message": None
            }
            
            return templates.TemplateResponse("main.html", context)
            
    except Exception as e:
        logger.error(f"Error in recommendations: {e}")
        context = {
            "request": request,
            "content_based_rec": pd.DataFrame(),
            "message": f"Error getting recommendations: {e}",
            "truncate": truncate,
            "random_product_image_urls": [],
            "random_price": random.choice(price_list)
        }
        return templates.TemplateResponse("main.html", context)

@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "data_loaded": trending_products is not None and train_data is not None,
        "trending_products_count": len(trending_products) if trending_products is not None else 0,
        "train_data_count": len(train_data) if train_data is not None else 0
    }

@app.get("/sample-products")
async def get_sample_products():
    """Get sample product names for testing recommendations"""
    if train_data is not None:
        sample_names = train_data['Name'].dropna().head(20).tolist()
        return {
            "status": "success",
            "sample_products": sample_names,
            "total_products": len(train_data['Name'].dropna()),
            "message": "Use these product names to test recommendations"
        }
    return {"status": "error", "message": "Training data not available"}

@app.post("/logout")
async def logout(request: Request, current_user: str = Depends(get_current_user)):
    """Logout user and clear session"""
    try:
        session_id = request.cookies.get("session_id")
        
        if session_id:
            logout_user(session_id)
        
        # Redirect to homepage after logout
        response = RedirectResponse(url="/", status_code=302)
        response.delete_cookie("session_id")
        
        return response
        
    except Exception as e:
        logger.error(f"Error during logout: {e}")
        response = RedirectResponse(url="/", status_code=302)
        response.delete_cookie("session_id")
        return response

@app.get("/user-info")
async def get_user_info(current_user: str = Depends(get_current_user)):
    """Get current user information"""
    if current_user:
        click_count = len(user_clicks.get(current_user, []))
        return {
            "status": "success",
            "user": current_user,
            "logged_in": True,
            "total_clicks": click_count,
            "message": f"Welcome back, {current_user}!"
        }
    else:
        return {
            "status": "info",
            "logged_in": False,
            "message": "Please login to track your preferences"
        }

@app.post("/track-click")
async def track_click(
    request: Request,
    product_name: str = Form(...),
    current_user: str = Depends(get_current_user)
):
    """Track user clicks on products (requires login)"""
    try:
        if not current_user:
            return {
                "status": "error", 
                "message": "Please login to track your preferences",
                "redirect": "login_required"
            }
        
        # Filter out invalid product names
        invalid_names = ['Sign In', 'Sign Up', 'Home', 'Settings', 'Google Form', '']
        if not product_name or product_name.strip() in invalid_names:
            return {
                "status": "error",
                "message": "Invalid product name"
            }
        
        global user_clicks
        
        if current_user not in user_clicks:
            user_clicks[current_user] = []
        
        # Add product to user's click history (avoid duplicates but track frequency)
        user_clicks[current_user].append({
            "product_name": product_name,
            "timestamp": datetime.now().isoformat(),
            "click_count": user_clicks[current_user].count(product_name) + 1
        })
        
        logger.info(f"👆 User {current_user} clicked on: {product_name}")
        
        return {
            "status": "success", 
            "message": f"Click tracked for {product_name}",
            "user": current_user,
            "total_clicks": len(user_clicks[current_user])
        }
    except Exception as e:
        logger.error(f"Error tracking click: {e}")
        return {"status": "error", "message": "Failed to track click"}

@app.get("/my-recommendations")
async def get_my_recommendations(
    current_user: str = Depends(get_current_user),
    limit: int = 10
):
    """Get personalized recommendations based on logged-in user's click history"""
    try:
        if not current_user:
            return {
                "status": "error",
                "message": "Please login to get personalized recommendations",
                "recommendations": []
            }
        
        if current_user not in user_clicks or not user_clicks[current_user]:
            return {
                "status": "info",
                "message": f"Hi {current_user}! Start browsing products to get personalized recommendations.",
                "recommendations": [],
                "user": current_user
            }
        
        # Get user click history
        user_click_history = user_clicks[current_user]
        recent_clicks = [click["product_name"] for click in user_click_history[-5:]]  # Last 5 clicks
        all_clicked_products = [click["product_name"] for click in user_click_history]
        
        # Get diverse recommendations based on multiple clicked products
        all_recommendations = []
        products_to_analyze = recent_clicks[-3:] if len(recent_clicks) >= 3 else recent_clicks  # Use up to 3 recent clicks
        
        for clicked_product in products_to_analyze:
            # Get recommendations for each clicked product
            product_recs = content_based_recommendations(train_data, clicked_product, top_n=limit*2)
            if not product_recs.empty:
                # Filter out NaN similarity and add source product info
                product_recs = product_recs[product_recs['similarity'].notna()]
                product_recs['source_click'] = clicked_product
                all_recommendations.append(product_recs)
        
        if not all_recommendations:
            last_clicked_product = recent_clicks[-1]
            recommendations = content_based_recommendations(train_data, last_clicked_product, top_n=limit)
            recommendations = recommendations[recommendations['similarity'].notna()]
        else:
            # Combine all recommendations
            import pandas as pd
            combined_recs = pd.concat(all_recommendations, ignore_index=True)
            
            # Remove products the user has already clicked
            combined_recs = combined_recs[~combined_recs['Name'].isin(all_clicked_products)]
            
            # Remove duplicates and keep the one with highest similarity
            recommendations = combined_recs.groupby('Name').agg({
                'Brand': 'first',
                'Category': 'first', 
                'Rating': 'first',
                'similarity': 'max',
                'source_click': 'first'
            }).reset_index()
            
            # Sort by similarity and take top recommendations
            recommendations = recommendations.sort_values('similarity', ascending=False).head(limit)
        
        if recommendations.empty:
            return {
                "status": "info", 
                "message": f"No new recommendations found based on your clicks. Try browsing more diverse products!",
                "recommendations": [],
                "user": current_user,
                "recent_clicks": recent_clicks,
                "total_clicks": len(user_click_history)
            }
        
        rec_list = recommendations.to_dict('records')
        
        # Generate LLM explanations for each recommendation
        for rec in rec_list:
            try:
                explanation = await generate_recommendation_explanation(
                    rec, 
                    user_click_history, 
                    rec['similarity']
                )
                rec['explanation'] = explanation
            except Exception as e:
                logger.error(f"Error generating explanation: {e}")
                rec['explanation'] = f"Recommended because you showed interest in similar products. {rec['similarity']:.1%} match!"
        
        return {
            "status": "success",
            "message": f"Hi {current_user}! Here are personalized recommendations based on your browsing history ({len(user_click_history)} clicks)",
            "user": current_user,
            "recent_clicks": recent_clicks,
            "total_clicks": len(user_click_history),
            "recommendations": rec_list,
            "analyzed_products": len(products_to_analyze) if 'products_to_analyze' in locals() else 1,
            "llm_powered": LLM_AVAILABLE
        }
        
    except Exception as e:
        logger.error(f"Error getting personalized recommendations: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return {
            "status": "error", 
            "message": "Failed to get recommendations",
            "recommendations": [],
            "user": current_user if current_user else "unknown"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)