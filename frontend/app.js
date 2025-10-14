// JavaScript for E-commerce Recommendation System Frontend

// API Configuration
const API_BASE_URL = 'http://localhost:8000';

// Global State
let currentPage = 1;
let currentProducts = [];
let selectedProductId = null;
const PRODUCTS_PER_PAGE = 12;

// Utility Functions
function showLoading() {
    document.getElementById('loadingSpinner').classList.remove('d-none');
}

function hideLoading() {
    document.getElementById('loadingSpinner').classList.add('d-none');
}

function showError(message) {
    const errorAlert = document.getElementById('errorAlert');
    const errorMessage = document.getElementById('errorMessage');
    errorMessage.textContent = message;
    errorAlert.classList.remove('d-none');
    setTimeout(() => errorAlert.classList.add('d-none'), 5000);
}

function showSuccess(message) {
    const successAlert = document.getElementById('successAlert');
    const successMessage = document.getElementById('successMessage');
    successMessage.textContent = message;
    successAlert.classList.remove('d-none');
    setTimeout(() => successAlert.classList.add('d-none'), 3000);
}

function hideAlerts() {
    document.getElementById('errorAlert').classList.add('d-none');
    document.getElementById('successAlert').classList.add('d-none');
}

// API Functions
async function apiCall(endpoint, options = {}) {
    try {
        const response = await fetch(`${API_BASE_URL}${endpoint}`, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API call failed:', error);
        throw error;
    }
}

// Product Display Functions
function createProductCard(product, showActions = true) {
    const rating = product.Rating || 0;
    const reviewCount = product.ReviewCount || 0;
    const stars = generateStars(rating);
    
    return `
        <div class="col-md-4 col-lg-3 mb-4">
            <div class="card product-card fade-in">
                <div class="product-image-placeholder">
                    <i class="fas fa-box"></i>
                </div>
                <div class="card-body">
                    <h6 class="product-title">${product.Name || 'Unnamed Product'}</h6>
                    <p class="product-brand text-muted">${product.Brand || 'Unknown Brand'}</p>
                    <div class="product-rating">
                        <span class="rating-stars">${stars}</span>
                        <span class="rating-text">${rating.toFixed(1)} (${reviewCount})</span>
                    </div>
                    ${showActions ? `
                    <div class="product-actions">
                        <button class="btn btn-outline-primary btn-sm" onclick="viewProduct(${product.ProdID})">
                            <i class="fas fa-eye me-1"></i>View
                        </button>
                        <button class="btn btn-success btn-sm" onclick="getProductRecommendations(${product.ProdID})">
                            <i class="fas fa-magic me-1"></i>Recommend
                        </button>
                    </div>
                    ` : ''}
                </div>
            </div>
        </div>
    `;
}

function createRecommendationCard(product, explanation = null) {
    const rating = product.Rating || 0;
    const reviewCount = product.ReviewCount || 0;
    const stars = generateStars(rating);
    
    return `
        <div class="recommendation-card slide-in">
            <div class="row">
                <div class="col-md-3">
                    <div class="product-image-placeholder">
                        <i class="fas fa-box"></i>
                    </div>
                </div>
                <div class="col-md-9">
                    <h5>${product.Name || 'Unnamed Product'}</h5>
                    <p class="text-muted mb-2">${product.Brand || 'Unknown Brand'}</p>
                    <div class="product-rating mb-3">
                        <span class="rating-stars">${stars}</span>
                        <span class="rating-text">${rating.toFixed(1)} (${reviewCount} reviews)</span>
                    </div>
                    ${explanation ? `
                    <div class="recommendation-explanation">
                        <i class="fas fa-lightbulb explanation-icon"></i>
                        <strong>Why this recommendation:</strong>
                        <p class="mb-0 mt-2">${explanation}</p>
                    </div>
                    ` : ''}
                    <div class="mt-3">
                        <button class="btn btn-outline-primary me-2" onclick="viewProduct(${product.ProdID})">
                            <i class="fas fa-eye me-1"></i>View Details
                        </button>
                        <button class="btn btn-success" onclick="getProductRecommendations(${product.ProdID})">
                            <i class="fas fa-magic me-1"></i>More Like This
                        </button>
                    </div>
                </div>
            </div>
        </div>
    `;
}

function generateStars(rating) {
    const fullStars = Math.floor(rating);
    const hasHalfStar = rating % 1 >= 0.5;
    const emptyStars = 5 - fullStars - (hasHalfStar ? 1 : 0);
    
    return '★'.repeat(fullStars) + 
           (hasHalfStar ? '☆' : '') + 
           '☆'.repeat(emptyStars);
}

// Main Loading Functions
async function loadTrendingProducts() {
    try {
        showLoading();
        hideAlerts();
        
        const products = await apiCall('/trending?limit=12');
        
        const container = document.getElementById('trendingProducts');
        if (products.length === 0) {
            container.innerHTML = '<div class="col-12 text-center"><p class="text-muted">No trending products found.</p></div>';
        } else {
            container.innerHTML = products.map(product => createProductCard(product)).join('');
        }
        
        // Populate product select dropdown
        populateProductSelect(products);
        
    } catch (error) {
        console.error('Error loading trending products:', error);
        showError('Failed to load trending products. Please try again.');
    } finally {
        hideLoading();
    }
}

async function loadAllProducts(page = 1) {
    try {
        showLoading();
        hideAlerts();
        
        const offset = (page - 1) * PRODUCTS_PER_PAGE;
        const products = await apiCall(`/products?limit=${PRODUCTS_PER_PAGE}&offset=${offset}`);
        
        const container = document.getElementById('allProducts');
        if (products.length === 0) {
            container.innerHTML = '<div class="col-12 text-center"><p class="text-muted">No products found.</p></div>';
        } else {
            container.innerHTML = products.map(product => createProductCard(product)).join('');
            currentProducts = products;
        }
        
        // Update pagination
        document.getElementById('pageInfo').textContent = `Page ${page}`;
        currentPage = page;
        
    } catch (error) {
        console.error('Error loading products:', error);
        showError('Failed to load products. Please try again.');
    } finally {
        hideLoading();
    }
}

async function populateProductSelect(products = null) {
    try {
        if (!products) {
            products = await apiCall('/products?limit=100');
        }
        
        const select = document.getElementById('productSelect');
        select.innerHTML = '<option value="">Choose a product...</option>';
        
        products.forEach(product => {
            const option = document.createElement('option');
            option.value = product.ProdID;
            option.textContent = `${product.Name} - ${product.Brand}`;
            select.appendChild(option);
        });
        
    } catch (error) {
        console.error('Error populating product select:', error);
    }
}

// Search Function
async function searchProducts() {
    const query = document.getElementById('searchInput').value.trim();
    if (!query) {
        showError('Please enter a search term.');
        return;
    }
    
    try {
        showLoading();
        hideAlerts();
        
        const products = await apiCall(`/search?query=${encodeURIComponent(query)}&limit=20`);
        
        // Switch to products tab and show results
        const productsTab = new bootstrap.Tab(document.getElementById('products-tab'));
        productsTab.show();
        
        const container = document.getElementById('allProducts');
        if (products.length === 0) {
            container.innerHTML = '<div class="col-12 text-center"><p class="text-muted">No products found for your search.</p></div>';
        } else {
            container.innerHTML = products.map(product => createProductCard(product)).join('');
            showSuccess(`Found ${products.length} products matching "${query}"`);
        }
        
    } catch (error) {
        console.error('Error searching products:', error);
        showError('Search failed. Please try again.');
    } finally {
        hideLoading();
    }
}

// Product Detail Functions
async function viewProduct(productId) {
    try {
        showLoading();
        
        const product = await apiCall(`/product/${productId}`);
        selectedProductId = productId;
        
        // Show product details in modal
        const modal = new bootstrap.Modal(document.getElementById('productModal'));
        document.getElementById('modalProductName').textContent = product.Name;
        
        const rating = product.Rating || 0;
        const reviewCount = product.ReviewCount || 0;
        const stars = generateStars(rating);
        
        document.getElementById('modalProductContent').innerHTML = `
            <div class="row">
                <div class="col-md-4">
                    <div class="product-image-placeholder">
                        <i class="fas fa-box"></i>
                    </div>
                </div>
                <div class="col-md-8">
                    <h4>${product.Name}</h4>
                    <p class="text-muted">${product.Brand}</p>
                    <div class="product-rating mb-3">
                        <span class="rating-stars">${stars}</span>
                        <span class="rating-text">${rating.toFixed(1)} (${reviewCount} reviews)</span>
                    </div>
                    <p><strong>Product ID:</strong> ${product.ProdID}</p>
                    <p><strong>Category:</strong> ${product.Category || 'N/A'}</p>
                    <p><strong>Description:</strong> ${product.Description || 'No description available'}</p>
                </div>
            </div>
        `;
        
        modal.show();
        
    } catch (error) {
        console.error('Error loading product details:', error);
        showError('Failed to load product details.');
    } finally {
        hideLoading();
    }
}

// Recommendation Functions
async function getRecommendations() {
    const productSelect = document.getElementById('productSelect');
    const recommendationType = document.getElementById('recommendationType').value;
    
    if (!productSelect.value) {
        showError('Please select a product first.');
        return;
    }
    
    const productId = productSelect.value;
    selectedProductId = productId;
    
    try {
        showLoading();
        hideAlerts();
        
        let recommendations;
        let endpoint;
        
        if (recommendationType === 'hybrid') {
            // Use a default user ID for hybrid recommendations
            const userId = 4;
            endpoint = `/recommend/hybrid/${productId}/${userId}/explained`;
        } else {
            endpoint = `/recommend/content/${productId}/explained`;
        }
        
        recommendations = await apiCall(endpoint);
        
        const container = document.getElementById('recommendationResults');
        
        if (recommendations.length === 0) {
            container.innerHTML = '<div class="text-center"><p class="text-muted">No recommendations found for this product.</p></div>';
        } else {
            const selectedProduct = await apiCall(`/product/${productId}`);
            
            container.innerHTML = `
                <div class="mb-4">
                    <h5><i class="fas fa-star text-warning me-2"></i>Recommendations for: ${selectedProduct.Name}</h5>
                    <p class="text-muted">Based on ${recommendationType === 'hybrid' ? 'AI analysis and user behavior' : 'product similarity'}</p>
                </div>
                ${recommendations.map(rec => createRecommendationCard(rec, rec.explanation)).join('')}
            `;
            
            showSuccess(`Found ${recommendations.length} recommendations!`);
        }
        
        // Switch to recommendations tab
        const recTab = new bootstrap.Tab(document.getElementById('recommendations-tab'));
        recTab.show();
        
    } catch (error) {
        console.error('Error getting recommendations:', error);
        showError('Failed to get recommendations. Please try again.');
    } finally {
        hideLoading();
    }
}

async function getProductRecommendations(productId = null) {
    if (!productId) {
        productId = selectedProductId;
    }
    
    if (!productId) {
        showError('No product selected.');
        return;
    }
    
    // Close modal if open
    const modal = bootstrap.Modal.getInstance(document.getElementById('productModal'));
    if (modal) {
        modal.hide();
    }
    
    // Set the product in the select dropdown
    document.getElementById('productSelect').value = productId;
    
    // Get recommendations
    await getRecommendations();
}

// Pagination Functions
function loadNextPage() {
    loadAllProducts(currentPage + 1);
}

function loadPreviousPage() {
    if (currentPage > 1) {
        loadAllProducts(currentPage - 1);
    }
}

// Event Listeners
document.addEventListener('DOMContentLoaded', function() {
    // Load trending products on page load
    loadTrendingProducts();
    
    // Load product select options
    populateProductSelect();
    
    // Add search functionality on Enter key
    document.getElementById('searchInput').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            searchProducts();
        }
    });
    
    // Tab change event listeners
    document.getElementById('products-tab').addEventListener('shown.bs.tab', function() {
        if (currentProducts.length === 0) {
            loadAllProducts(1);
        }
    });
    
    document.getElementById('recommendations-tab').addEventListener('shown.bs.tab', function() {
        if (document.getElementById('productSelect').children.length <= 1) {
            populateProductSelect();
        }
    });
});

// Error handling for network issues
window.addEventListener('online', function() {
    showSuccess('Connection restored!');
});

window.addEventListener('offline', function() {
    showError('You are offline. Some features may not work.');
});