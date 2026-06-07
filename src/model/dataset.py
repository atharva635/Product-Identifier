import os
import json
import random
import requests
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset
from transformers import CLIPProcessor

# High-quality real products database mapping to creative commons Unsplash images
REAL_PRODUCTS = [
    # Footwear
    {
        "name": "Air Zoom Running Shoes",
        "brand": "Nike",
        "category": "Footwear",
        "price": 8999,
        "color": "red",
        "description": "High-performance red athletic sneakers featuring responsive cushioning, breathable mesh upper, and durable rubber outsoles. Perfect for marathons, gym workouts, and jogging.",
        "tags": ["running", "sport", "cushioning", "marathon", "gym", "lightweight"],
        "url": "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=400&q=80"
    },
    {
        "name": "Classic Street Sneakers",
        "brand": "Puma",
        "category": "Footwear",
        "price": 4999,
        "color": "black",
        "description": "Classic black leather lifestyle sneakers with soft cushioned sockliner and retro vulcanized rubber sole. Elegant and durable daily footwear.",
        "tags": ["casual", "retro", "leather", "daily", "lifestyle"],
        "url": "https://images.unsplash.com/photo-1539185441755-769473a23570?w=400&q=80"
    },
    {
        "name": "Ultraboost Training Shoes",
        "brand": "Adidas",
        "category": "Footwear",
        "price": 12999,
        "color": "blue",
        "description": "Premium blue lightweight training shoes with energy-returning boost midsole and Primeknit upper. Provides exceptional comfort and grip for heavy workout sessions.",
        "tags": ["training", "boost", "comfort", "gym", "workout", "lightweight"],
        "url": "https://images.unsplash.com/photo-1460353581641-37baddab0fa2?w=400&q=80"
    },
    {
        "name": "Heritage Leather Boots",
        "brand": "Timberland",
        "category": "Footwear",
        "price": 14999,
        "color": "brown",
        "description": "Heavy-duty waterproof brown leather boots with seam-sealed construction, rustproof hardware, and rubber lug outsoles for superior outdoor traction.",
        "tags": ["boots", "waterproof", "leather", "outdoor", "traction", "hiking"],
        "url": "https://images.unsplash.com/photo-1520639888713-7851133b1ed0?w=400&q=80"
    },
    {
        "name": "Classic 1460 Combat Boots",
        "brand": "Dr. Martens",
        "category": "Footwear",
        "price": 11999,
        "color": "black",
        "description": "Iconic 8-eye black leather combat boots with yellow welt stitching and air-cushioned slip-resistant sole. Extremely robust and retro.",
        "tags": ["boots", "combat", "leather", "retro", "classic", "grunge"],
        "url": "https://images.unsplash.com/photo-1608256246200-53e635b5b65f?w=400&q=80"
    },
    {
        "name": "Arizona Two-Strap Sandals",
        "brand": "Birkenstock",
        "category": "Footwear",
        "price": 6999,
        "color": "brown",
        "description": "Classic brown suede double-strap sandals with adjustable metal buckles and contoured cork-latex footbed that conforms to the shape of your foot.",
        "tags": ["sandals", "cork", "casual", "summer", "comfort"],
        "url": "https://images.unsplash.com/photo-1603487742131-4160ec999306?w=400&q=80"
    },
    {
        "name": "Adilette Sport Slides",
        "brand": "Adidas",
        "category": "Footwear",
        "price": 2499,
        "color": "black",
        "description": "Comfortable black slide sandals with quick-dry bandage upper, signature 3-stripes, and contoured footbed for indoor and pool-side usage.",
        "tags": ["sandals", "slides", "pool", "dry", "sport", "casual"],
        "url": "https://images.unsplash.com/photo-1605980776566-0486c3ac7617?w=400&q=80"
    },
    # Fashion
    {
        "name": "Classic Caviar Handbag",
        "brand": "Chanel",
        "category": "Fashion",
        "price": 85000,
        "color": "red",
        "description": "Stunning red luxury handbag crafted with quilted caviar leather, gold-tone chain link shoulder strap, and iconic CC turn-lock closure.",
        "tags": ["handbag", "luxury", "leather", "quilted", "gold", "premium"],
        "url": "https://images.unsplash.com/photo-1584917865442-de89df76afd3?w=400&q=80"
    },
    {
        "name": "Marmont Shoulder Bag",
        "brand": "Gucci",
        "category": "Fashion",
        "price": 74999,
        "color": "brown",
        "description": "Elegant brown leather shoulder bag featuring chevron design quilting, interlocking antique gold-tone GG hardware, and zip top closure.",
        "tags": ["handbag", "luxury", "leather", "quilted", "gold", "designer"],
        "url": "https://images.unsplash.com/photo-1590874103328-eac38a683ce7?w=400&q=80"
    },
    {
        "name": "Heritage Travel Backpack",
        "brand": "Herschel",
        "category": "Fashion",
        "price": 5499,
        "color": "brown",
        "description": "Classic brown canvas travel backpack with signature striped fabric liner, 15-inch laptop sleeve, front pocket, and synthetic leather details.",
        "tags": ["backpack", "canvas", "travel", "laptop", "campus", "vintage"],
        "url": "https://images.unsplash.com/photo-1553062407-98eeb64c6a62?w=400&q=80"
    },
    {
        "name": "Urban Explorer Backpack",
        "brand": "North Face",
        "category": "Fashion",
        "price": 6999,
        "color": "black",
        "description": "Water-resistant black tactical commuter backpack with compression straps, mesh side pockets, ergonomic shoulder straps, and modular laptop compartment.",
        "tags": ["backpack", "tactical", "waterproof", "laptop", "commute", "outdoor"],
        "url": "https://images.unsplash.com/photo-1581605405669-fcdf81165afa?w=400&q=80"
    },
    {
        "name": "Premium Leather Jacket",
        "brand": "Schott NYC",
        "category": "Fashion",
        "price": 29999,
        "color": "black",
        "description": "Heavyweight black steerhide leather motorcycle jacket with asymmetrical front zipper, snap-down collar, three zippered pockets, and belt detail.",
        "tags": ["jacket", "leather", "biker", "premium", "retro", "classic"],
        "url": "https://images.unsplash.com/photo-1551028719-00167b16eac5?w=400&q=80"
    },
    {
        "name": "Sherpa Denim Trucker Jacket",
        "brand": "Levi's",
        "category": "Fashion",
        "price": 4999,
        "color": "blue",
        "description": "Classic blue cotton denim trucker jacket lined with warm, cozy faux-sherpa lining. Features snap buttons and chest flap pockets.",
        "tags": ["jacket", "denim", "sherpa", "casual", "winter", "blue"],
        "url": "https://images.unsplash.com/photo-1576995853123-5a10305d93c0?w=400&q=80"
    },
    # Electronics
    {
        "name": "MacBook Pro 14 M3",
        "brand": "Apple",
        "category": "Electronics",
        "price": 169900,
        "color": "silver",
        "description": "Powerful silver notebook laptop featuring the Apple M3 chip, 14.2-inch Liquid Retina XDR display, 8GB RAM, and 512GB SSD. Outstanding battery life and speed.",
        "tags": ["laptop", "apple", "m3", "retina", "silver", "premium"],
        "url": "https://images.unsplash.com/photo-1517336714731-489689fd1ca8?w=400&q=80"
    },
    {
        "name": "ThinkPad X1 Carbon Gen 11",
        "brand": "Lenovo",
        "category": "Electronics",
        "price": 144999,
        "color": "black",
        "description": "Ultralight black business laptop with carbon-fiber chassis, Intel Core i7, 14-inch anti-glare screen, backlit keyboard, and fingerprint reader.",
        "tags": ["laptop", "lenovo", "thinkpad", "business", "intel", "lightweight"],
        "url": "https://images.unsplash.com/photo-1588872657578-7efd1f1555ed?w=400&q=80"
    },
    {
        "name": "iPhone 15 Pro Max",
        "brand": "Apple",
        "category": "Electronics",
        "price": 139900,
        "color": "pink",
        "description": "Flagship Apple smartphone in titanium pink, featuring a 6.7-inch OLED Super Retina XDR display, A17 Pro chip, triple-camera system, and USB-C.",
        "tags": ["smartphone", "apple", "iphone", "camera", "pink", "oled"],
        "url": "https://images.unsplash.com/photo-1511707171634-5f897ff02aa9?w=400&q=80"
    },
    {
        "name": "Galaxy S24 Ultra",
        "brand": "Samsung",
        "category": "Electronics",
        "price": 119900,
        "color": "black",
        "description": "Premium Android smartphone in titanium black, featuring 6.8-inch Dynamic AMOLED 120Hz display, 200MP camera, built-in S-Pen, and Snapdragon 8 Gen 3.",
        "tags": ["smartphone", "samsung", "android", "spen", "camera", "amoled"],
        "url": "https://images.unsplash.com/photo-1598327105666-5b89351aff97?w=400&q=80"
    },
    {
        "name": "WH-1000XM5 ANC Headphones",
        "brand": "Sony",
        "category": "Electronics",
        "price": 26990,
        "color": "black",
        "description": "Industry-leading black over-ear wireless headphones with dual processor auto-ANC, 30 hours battery life, speak-to-chat, and crystal-clear call quality.",
        "tags": ["headphones", "anc", "wireless", "sony", "music", "bluetooth"],
        "url": "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=400&q=80"
    },
    {
        "name": "QuietComfort Headphones",
        "brand": "Bose",
        "category": "Electronics",
        "price": 22990,
        "color": "silver",
        "description": "Legendary noise-cancelling silver headphones with premium hi-fi audio quality, custom sound EQ profiles, and lightweight comfort cushion design.",
        "tags": ["headphones", "bose", "anc", "comfort", "silver", "wireless"],
        "url": "https://images.unsplash.com/photo-1546435770-a3e426bf472b?w=400&q=80"
    },
    # Accessories
    {
        "name": "Minimalist Chronograph Watch",
        "brand": "Fossil",
        "category": "Accessories",
        "price": 11995,
        "color": "silver",
        "description": "Clean silver-tone analog wristwatch featuring a white dial, sub-dials, quartz movement, mineral glass casing, and stainless steel mesh strap.",
        "tags": ["watch", "fossil", "analog", "quartz", "chronograph", "silver"],
        "url": "https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=400&q=80"
    },
    {
        "name": "5 Sports Automatic Watch",
        "brand": "Seiko",
        "category": "Accessories",
        "price": 24500,
        "color": "black",
        "description": "Sleek black automatic mechanical wristwatch with stainless steel case, unidirectional rotating bezel, day/date display, and 100m water resistance.",
        "tags": ["watch", "seiko", "automatic", "mechanical", "waterproof", "black"],
        "url": "https://images.unsplash.com/photo-1547996160-81dfa63595aa?w=400&q=80"
    },
    {
        "name": "Classic Wayfarer Sunglasses",
        "brand": "Ray-Ban",
        "category": "Accessories",
        "price": 9990,
        "color": "gold",
        "description": "Timeless G-15 green polarized lenses with gold-trimmed acetate frames. Features UV protection coating and comfortable nose grips.",
        "tags": ["sunglasses", "rayban", "classic", "polarized", "uv", "vintage"],
        "url": "https://images.unsplash.com/photo-1572635196237-14b3f281503f?w=400&q=80"
    },
    {
        "name": "Polarized Sport Sunglasses",
        "brand": "Oakley",
        "category": "Accessories",
        "price": 12990,
        "color": "black",
        "description": "Robust black matte sport sunglasses with polarized Prizm Road lenses, lightweight frame, and rubber temple guards for anti-slip grip.",
        "tags": ["sunglasses", "sport", "polarized", "oakley", "cycling", "run"],
        "url": "https://images.unsplash.com/photo-1511499767150-a48a237f0083?w=400&q=80"
    },
    {
        "name": "Saddleback Leather Bifold",
        "brand": "Saddleback",
        "category": "Accessories",
        "price": 4500,
        "color": "brown",
        "description": "Thick full-grain brown leather bifold wallet with RFID shielding, four card slots, and cash sleeve. Extremely durable with reinforced stitching.",
        "tags": ["wallet", "leather", "bifold", "rfid", "brown", "durable"],
        "url": "https://images.unsplash.com/photo-1627124765135-565701355a74?w=400&q=80"
    },
    {
        "name": "Slim Sleeve Leather Wallet",
        "brand": "Bellroy",
        "category": "Accessories",
        "price": 5990,
        "color": "black",
        "description": "Slim bifold black leather wallet with quick-access slots for active cards, a pull-tab section for less-used cards, and folded bill sleeve.",
        "tags": ["wallet", "leather", "slim", "bellroy", "minimalist", "black"],
        "url": "https://images.unsplash.com/photo-1627124769350-b0ea8b4e723d?w=400&q=80"
    }
]

def draw_fallback_image(shape_color, text_label, output_path):
    """Draws a placeholder colored image in case Unsplash download fails."""
    img = Image.new("RGB", (224, 224), (243, 244, 246))
    draw = ImageDraw.Draw(img)
    # Draw circle
    draw.ellipse([20, 20, 204, 204], fill=(255, 255, 255), outline=(229, 231, 235), width=2)
    # Draw colored square
    draw.rectangle([60, 60, 164, 164], fill=shape_color)
    # Draw simple label line
    draw.line([(60, 164), (164, 164)], fill=(180, 180, 180), width=4)
    img.save(output_path, "JPEG", quality=95)

def generate_synthetic_catalog(output_dir="data", count_per_template=1):
    """Generates the catalog JSON and downloads high-quality product images."""
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    catalog = []
    prod_id_counter = 1

    # Standard HSL mapped colors for fallbacks
    fallback_colors = {
        "red": (239, 68, 68),
        "blue": (59, 130, 246),
        "green": (34, 197, 94),
        "yellow": (234, 179, 8),
        "black": (31, 41, 55),
        "brown": (120, 53, 4),
        "pink": (236, 72, 153),
        "orange": (249, 115, 22),
        "purple": (147, 51, 234),
        "silver": (156, 163, 175),
        "gold": (217, 119, 6)
    }

    # Headers for Unsplash download request
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    print("Beginning seeding of real product images and catalog metadata...")

    for item in REAL_PRODUCTS:
        prod_id = f"prod_{prod_id_counter:03d}"
        title = f"{item['brand']} {item['name']}"
        
        image_filename = f"{prod_id}.jpg"
        image_path = os.path.join("data", "images", image_filename)
        full_image_path = os.path.join(output_dir, "images", image_filename)
        
        # Download image from Unsplash
        download_success = False
        try:
            response = requests.get(item["url"], headers=headers, timeout=10)
            if response.status_code == 200:
                with open(full_image_path, "wb") as f:
                    f.write(response.content)
                # Verify image can be opened
                with Image.open(full_image_path) as img:
                    img.verify()
                download_success = True
                print(f"[{prod_id}] Successfully downloaded product image for: {title}")
        except Exception as e:
            print(f"[{prod_id}] Failed to download/verify {title} image: {e}")

        # Fallback to drawing a stylized square if download failed
        if not download_success:
            print(f"[{prod_id}] Generating fallback graphic placeholder for: {title}")
            rgb_color = fallback_colors.get(item["color"], (100, 100, 100))
            draw_fallback_image(rgb_color, item["name"], full_image_path)

        catalog.append({
            "id": prod_id,
            "title": title,
            "brand": item["brand"],
            "category": item["category"],
            "price": item["price"],
            "color": item["color"],
            "description": item["description"],
            "tags": item["tags"],
            "image_path": image_path
        })
        
        prod_id_counter += 1

    # Save catalog.json
    with open(os.path.join(output_dir, "catalog.json"), "w") as f:
        json.dump(catalog, f, indent=4)
        
    print(f"Generated product catalog containing {len(catalog)} products and images in '{output_dir}'.")
    return catalog

class ECommerceDataset(Dataset):
    """Custom PyTorch Dataset for loading e-commerce image-text pairs."""
    def __init__(self, catalog_path="data/catalog.json", processor_name="openai/clip-vit-base-patch32"):
        # Load catalog
        with open(catalog_path, "r") as f:
            self.catalog = json.load(f)
            
        # Initialize the CLIP processor
        self.processor = CLIPProcessor.from_pretrained(processor_name)
        
        # Base dir for image loading
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(catalog_path)))

    def __len__(self):
        return len(self.catalog)

    def __getitem__(self, idx):
        item = self.catalog[idx]
        
        # Load image
        img_path = os.path.join(self.base_dir, item["image_path"])
        image = Image.open(img_path).convert("RGB")
        
        # We align Image and Text
        # The text can be the title + description combined for rich semantics
        text_content = f"{item['title']} - {item['category']} by {item['brand']}: {item['description']} tags: {', '.join(item['tags'])}"
        
        # Process inputs for CLIP
        processed = self.processor(
            text=[text_content],
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77
        )
        
        # Remove batch dimension from processor outputs since DataLoader will add it back
        return {
            "input_ids": processed["input_ids"].squeeze(0),
            "attention_mask": processed["attention_mask"].squeeze(0),
            "pixel_values": processed["pixel_values"].squeeze(0),
            "id": item["id"]
        }

if __name__ == "__main__":
    generate_synthetic_catalog()
