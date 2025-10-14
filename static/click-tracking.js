/* Click Tracking for Product Recommendations */

// Function to track product clicks (requires login)
async function trackProductClick(productName) {
    try {
        // Check if user is logged in first
        const userInfo = await fetch('/user-info');
        const userData = await userInfo.json();
        
        if (!userData.logged_in) {
            showLoginPrompt();
            return;
        }
        
        const formData = new FormData();
        formData.append('product_name', productName);
        
        const response = await fetch('/track-click', {
            method: 'POST',
            body: formData,
            credentials: 'include'  // Include cookies
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            console.log(`✅ Tracked click on: ${productName} for user: ${result.user}`);
            
            // Show notification
            showClickNotification(productName, result.user);
            
            // Update recommendations based on clicks
            updatePersonalRecommendations();
        } else if (result.redirect === 'login_required') {
            showLoginPrompt();
        }
    } catch (error) {
        console.error('Error tracking click:', error);
    }
}

// Function to show login prompt
function showLoginPrompt() {
    const notification = document.createElement('div');
    notification.className = 'login-prompt';
    notification.innerHTML = `
        <div class="alert alert-warning alert-dismissible fade show" role="alert" style="position: fixed; top: 20px; right: 20px; z-index: 1050; max-width: 350px;">
            <i class="fas fa-sign-in-alt"></i> Please login to track your preferences and get personalized recommendations!
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;
    
    document.body.appendChild(notification);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        notification.remove();
    }, 5000);
}

// Function to show click notification
function showClickNotification(productName, username) {
    // Create a small notification
    const notification = document.createElement('div');
    notification.className = 'click-notification';
    notification.innerHTML = `
        <div class="alert alert-success alert-dismissible fade show" role="alert" style="position: fixed; top: 20px; right: 20px; z-index: 1050; max-width: 350px;">
            <i class="fas fa-user-check"></i> Hi ${username}! Click tracked for "${productName}". Getting your recommendations...
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;
    
    document.body.appendChild(notification);
    
    // Auto-remove after 4 seconds
    setTimeout(() => {
        notification.remove();
    }, 4000);
}

// Function to get personalized recommendations based on clicks
async function updatePersonalRecommendations() {
    try {
        const response = await fetch('/my-recommendations', {
            credentials: 'include'  // Include cookies
        });
        const result = await response.json();
        
        if (result.status === 'success' && result.recommendations.length > 0) {
            displayPersonalRecommendations(result);
        } else if (result.status === 'info' && result.message) {
            // Show info message for users who haven't clicked enough products
            const container = document.getElementById('personal-recommendations');
            if (container) {
                container.innerHTML = `
                    <div class="alert alert-info">
                        <h4><i class="fas fa-info-circle"></i> ${result.message}</h4>
                        <p>Click on products above to start getting personalized recommendations!</p>
                    </div>
                `;
            }
        }
    } catch (error) {
        console.error('Error getting personal recommendations:', error);
    }
}

// Function to display click-based recommendations
// Function to display personalized recommendations
function displayPersonalRecommendations(result) {
    const container = document.getElementById('personal-recommendations');
    if (!container) return;
    
    let html = `
        <div class="alert alert-success">
            <h4><i class="fas fa-star"></i> ${result.message}</h4>
            <div class="row">
    `;
    
    result.recommendations.forEach((rec, index) => {
        const explanation = rec.explanation || `Recommended based on ${(rec.similarity * 100).toFixed(1)}% similarity to your interests.`;
        
        html += `
            <div class="col-md-6 col-lg-4 mb-3">
                <div class="card border-success h-100">
                    <div class="card-body d-flex flex-column">
                        <h6 class="card-title text-success">${rec.Name || rec.Product}</h6>
                        <div class="mb-2">
                            <small class="text-muted">Brand: ${rec.Brand || 'N/A'}</small><br>
                            <small class="text-muted">Similarity: ${(rec.similarity * 100).toFixed(1)}%</small>
                        </div>
                        <div class="alert alert-light border-0 bg-light mb-3 flex-grow-1">
                            <small><i class="fas fa-lightbulb text-warning"></i> <strong>Why recommended:</strong><br>
                            ${explanation}</small>
                        </div>
                        <div class="mt-auto">
                            <button class="btn btn-sm btn-outline-success w-100" onclick="trackProductClick('${rec.Name || rec.Product}')">
                                <i class="fas fa-heart"></i> Interested
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;
    });
    
    html += `
            </div>
        </div>
    `;
    
    container.innerHTML = html;
}

// Add click tracking to existing product cards
document.addEventListener('DOMContentLoaded', function() {
    // Check if user is logged in and load recommendations
    setTimeout(() => {
        updatePersonalRecommendations();
    }, 1000);
    
    // Track clicks on clickable product elements (images and titles)
    document.querySelectorAll('.clickable-product').forEach(element => {
        element.addEventListener('click', function() {
            const productName = this.getAttribute('data-product-name');
            if (productName) {
                trackProductClick(productName);
            }
        });
    });
    
    // Track clicks on "Buy Now" buttons
    document.querySelectorAll('[data-toggle="modal"]').forEach(button => {
        button.addEventListener('click', function() {
            const modal = this.getAttribute('data-target');
            if (modal) {
                const productName = document.querySelector(modal + ' .modal-title')?.textContent;
                if (productName) {
                    trackProductClick(productName);
                }
            }
        });
    });
});

// Add CSS for better styling
const style = document.createElement('style');
style.textContent = `
    .recommendation-card {
        transition: transform 0.2s, box-shadow 0.2s;
        cursor: pointer;
    }
    
    .recommendation-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .click-notification {
        animation: slideIn 0.3s ease;
    }
    
    @keyframes slideIn {
        from { transform: translateX(100%); }
        to { transform: translateX(0); }
    }
`;
document.head.appendChild(style);