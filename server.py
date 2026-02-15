from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, File, Form, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field, EmailStr, ConfigDict
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dotenv import load_dotenv
import os
import uuid
import logging
import httpx

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =====================================================
# PYDANTIC MODELS
# =====================================================

# User Authentication Model
class User(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    user_id: str = Field(default_factory=lambda: f"user_{uuid.uuid4().hex[:12]}")
    email: EmailStr
    name: Optional[str] = None
    picture: Optional[str] = None
    password_hash: Optional[str] = None
    auth_provider: str = "email"  # email, google, facebook, whatsapp
    is_verified: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_login: Optional[datetime] = None


# User Session Model
class UserSession(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    user_id: str
    session_token: str
    expires_at: datetime
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# Doctor Registration Model
class DoctorProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    
    # Step 1: Basic Information
    title: str  # Dr, Prof, Mr, Ms
    first_name: str
    surname: str
    gender: str  # Male, Female
    date_of_birth: str
    photo_url: Optional[str] = None
    is_arya_vysya: Optional[bool] = None
    gotram: Optional[str] = None
    wants_to_be_moderator: bool = False
    
    # Step 2: Medical Stream & Credentials
    medical_stream: str  # MBBS, Dental, Ayurveda, Homeo, Veterinary
    current_position: Optional[str] = None
    degrees_completed: List[str] = []
    council_registration_number: str
    registered_council: str
    
    # Step 3: Contact & Location
    email: EmailStr
    mobile_number: str
    other_phone_numbers: Optional[str] = None
    state: Optional[str] = None
    city: str
    pincode: Optional[str] = None
    country: str = "India"
    home_address: Optional[str] = None
    office_address: Optional[str] = None
    
    # Step 4: Professional Details
    attached_to_teaching_institute: bool = False
    teaching_institute_details: Optional[str] = None
    areas_of_interest: Optional[str] = None
    specialty: Optional[str] = None
    sub_specialty: Optional[str] = None
    years_of_experience: Optional[int] = None
    practice_clinic_name: Optional[str] = None
    
    # Step 5: Optional Information
    ancestral_place: Optional[str] = None
    ug_year: Optional[str] = None
    ug_institute: Optional[str] = None
    pg_year: Optional[str] = None
    pg_institute: Optional[str] = None
    associations: Optional[str] = None
    spouse_name: Optional[str] = None
    spouse_profession: Optional[str] = None
    children_details: Optional[str] = None
    hobbies: Optional[str] = None
    is_vysya: Optional[bool] = None
    post_graduation_level: Optional[str] = None  # in UG, in PG
    
    # Step 6: Privacy & Consent
    display_phone_publicly: bool = True
    display_photo_publicly: bool = True
    consent_downloadable_directory: bool = False
    terms_accepted: bool = False
    
    # Admin fields
    membership_tier: str = "Premium"  # Essential, Premium, VIP
    is_early_bird: bool = False  # First 100 free Premium
    payment_status: str = "pending"  # pending, paid, free
    payment_amount: float = 0.0
    payment_date: Optional[datetime] = None
    tier_expires_at: Optional[datetime] = None  # Annual renewal
    status: str = "pending"  # pending, approved, rejected
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Profile analytics
    profile_views: int = 0
    last_viewed_at: Optional[datetime] = None


# Registration Input Models
class DoctorRegistrationStep1(BaseModel):
    title: str
    first_name: str
    surname: str
    gender: str
    date_of_birth: str


class DoctorRegistrationStep2(BaseModel):
    medical_stream: str
    degrees_completed: List[str]
    council_registration_number: str
    registered_council: str


class DoctorRegistrationStep3(BaseModel):
    email: EmailStr
    mobile_number: str
    other_phone_numbers: Optional[str] = None
    city: str
    state: Optional[str] = None
    country: str = "India"
    home_address: Optional[str] = None
    office_address: Optional[str] = None
    hospital_address: Optional[str] = None


class DoctorRegistrationStep4(BaseModel):
    attached_to_teaching_institute: bool = False
    teaching_institute_details: Optional[str] = None
    areas_of_interest: Optional[str] = None
    specialty: Optional[str] = None
    sub_specialty: Optional[str] = None
    years_of_experience: Optional[int] = None
    practice_clinic_name: Optional[str] = None


class DoctorRegistrationStep5(BaseModel):
    gotram: Optional[str] = None
    ancestral_place: Optional[str] = None
    ug_year: Optional[str] = None
    ug_institute: Optional[str] = None
    pg_year: Optional[str] = None
    pg_institute: Optional[str] = None
    associations: Optional[str] = None
    spouse_name: Optional[str] = None
    spouse_profession: Optional[str] = None
    children_details: Optional[str] = None
    hobbies: Optional[str] = None
    is_vysya: Optional[bool] = None
    post_graduation_level: Optional[str] = None


class DoctorRegistrationStep6(BaseModel):
    display_phone_publicly: bool = True
    display_photo_publicly: bool = True
    consent_downloadable_directory: bool = False
    terms_accepted: bool = True


# Search/Filter Model
class DoctorSearchFilters(BaseModel):
    medical_stream: Optional[str] = None
    specialty: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    gender: Optional[str] = None
    keyword: Optional[str] = None
    skip: int = 0
    limit: int = 20


# Chapter Model
class Chapter(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    type: str  # state, city, specialty
    state: Optional[str] = None
    city: Optional[str] = None
    specialty: Optional[str] = None
    moderator_id: Optional[str] = None
    moderator_name: Optional[str] = None
    members_count: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# Matrimony Profile Model
class MatrimonyProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doctor_id: str
    is_active: bool = True
    age: int
    height: Optional[str] = None
    education: Optional[str] = None
    occupation_details: Optional[str] = None
    family_details: Optional[str] = None
    expectations: Optional[str] = None
    photos: List[str] = []
    contact_preferences: Dict[str, Any] = {}
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# Advertisement Model
class Advertisement(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doctor_id: str
    plan: str  # Bronze, Silver, Gold, Platinum
    ad_type: str  # banner, video, featured
    content_url: str
    headline: Optional[str] = None
    description: Optional[str] = None
    cta_link: Optional[str] = None
    status: str = "pending"  # pending, approved, active, expired
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    views: int = 0
    clicks: int = 0
    amount_paid: float = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# =====================================================
# API ENDPOINTS
# =====================================================

@api_router.get("/")
async def root():
    return {"message": "AVDA API - Arya Vysya Doctors Alliance", "version": "1.0.0"}


@api_router.get("/health")
async def health_check():
    return {"status": "healthy", "database": "connected"}


# =====================================================
# AUTHENTICATION ENDPOINTS (Google OAuth)
# =====================================================

def get_user_from_cookie_or_header(request: Request) -> Optional[str]:
    """Get session token from cookie (priority) or Authorization header (fallback)"""
    # Try cookie first
    session_token = request.cookies.get("session_token")
    if session_token:
        return session_token
    
    # Fallback to Authorization header
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        return auth_header.replace("Bearer ", "")
    
    return None


@api_router.post("/auth/session")
async def create_session(request: Request, response: Response, session_id: str):
    """
    Exchange Emergent session_id for user data and create persistent session
    REMINDER: DO NOT HARDCODE THE URL, OR ADD ANY FALLBACKS OR REDIRECT URLS, THIS BREAKS THE AUTH
    """
    try:
        # Call Emergent Auth API to get user data
        async with httpx.AsyncClient() as client:
            emergent_response = await client.get(
                "https://demobackend.emergentagent.com/auth/v1/env/oauth/session-data",
                headers={"X-Session-ID": session_id},
                timeout=10.0
            )
            
            if emergent_response.status_code != 200:
                raise HTTPException(status_code=401, detail="Invalid session ID")
            
            user_data = emergent_response.json()
        
        # Check if user exists
        existing_user = await db.users.find_one({"email": user_data["email"]}, {"_id": 0})
        
        if existing_user:
            # Update existing user
            user_id = existing_user["user_id"]
            await db.users.update_one(
                {"user_id": user_id},
                {
                    "$set": {
                        "name": user_data.get("name"),
                        "picture": user_data.get("picture"),
                        "last_login": datetime.now(timezone.utc).isoformat()
                    }
                }
            )
        else:
            # Create new user
            user_id = f"user_{uuid.uuid4().hex[:12]}"
            await db.users.insert_one({
                "user_id": user_id,
                "email": user_data["email"],
                "name": user_data.get("name"),
                "picture": user_data.get("picture"),
                "auth_provider": "google",
                "is_verified": True,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "last_login": datetime.now(timezone.utc).isoformat()
            })
        
        # Create session
        session_token = user_data.get("session_token") or f"session_{uuid.uuid4().hex}"
        expires_at = datetime.now(timezone.utc) + timedelta(days=7)
        
        await db.user_sessions.insert_one({
            "user_id": user_id,
            "session_token": session_token,
            "expires_at": expires_at.isoformat(),
            "created_at": datetime.now(timezone.utc).isoformat()
        })
        
        # Set httpOnly cookie
        response.set_cookie(
            key="session_token",
            value=session_token,
            httponly=True,
            secure=True,
            samesite="none",
            path="/",
            max_age=7 * 24 * 60 * 60  # 7 days
        )
        
        # Return user data
        user = await db.users.find_one({"user_id": user_id}, {"_id": 0})
        return user
        
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Emergent Auth timeout")
    except Exception as e:
        logger.error(f"Session creation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/auth/me")
async def get_current_user(request: Request):
    """Get current authenticated user"""
    session_token = get_user_from_cookie_or_header(request)
    
    if not session_token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    # Find session
    session_doc = await db.user_sessions.find_one({"session_token": session_token}, {"_id": 0})
    
    if not session_doc:
        raise HTTPException(status_code=401, detail="Invalid session")
    
    # Check expiry
    expires_at = session_doc["expires_at"]
    if isinstance(expires_at, str):
        expires_at = datetime.fromisoformat(expires_at)
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    
    if expires_at < datetime.now(timezone.utc):
        # Clean up expired session
        await db.user_sessions.delete_one({"session_token": session_token})
        raise HTTPException(status_code=401, detail="Session expired")
    
    # Get user
    user = await db.users.find_one({"user_id": session_doc["user_id"]}, {"_id": 0})
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return user


@api_router.post("/auth/logout")
async def logout(request: Request, response: Response):
    """Logout user"""
    session_token = get_user_from_cookie_or_header(request)
    
    if session_token:
        # Delete session from database
        await db.user_sessions.delete_one({"session_token": session_token})
    
    # Clear cookie
    response.delete_cookie(key="session_token", path="/")
    
    return {"message": "Logged out successfully"}


# =====================================================
# DOCTOR REGISTRATION ENDPOINTS
# =====================================================

@api_router.post("/doctors/register", response_model=DoctorProfile)
async def register_doctor(profile: DoctorProfile, request: Request):
    """Complete doctor registration - requires authentication"""
    try:
        # Get authenticated user
        session_token = get_user_from_cookie_or_header(request)
        if not session_token:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        session_doc = await db.user_sessions.find_one({"session_token": session_token}, {"_id": 0})
        if not session_doc:
            raise HTTPException(status_code=401, detail="Invalid session")
        
        # Use authenticated user's ID
        profile.user_id = session_doc["user_id"]
        
        # Check if doctor already registered
        existing = await db.doctors.find_one({"user_id": profile.user_id}, {"_id": 0})
        if existing:
            raise HTTPException(status_code=400, detail="Doctor profile already exists for this user")
        
        # Check total approved doctors for early bird status
        total_approved = await db.doctors.count_documents({"status": "approved"})
        
        # First 100 get Premium tier FREE
        if total_approved < 100:
            profile.membership_tier = "Premium"
            profile.is_early_bird = True
            profile.payment_status = "free"
            profile.payment_amount = 0.0
            profile.tier_expires_at = datetime.now(timezone.utc) + timedelta(days=365)  # 1 year
            logger.info(f"Early bird registration #{total_approved + 1} - Premium tier granted free")
        else:
            # After 100, use selected tier (default Premium, will need payment)
            if not profile.membership_tier:
                profile.membership_tier = "Premium"
            profile.payment_status = "pending"
            # Set payment amount based on tier
            tier_prices = {"Essential": 100, "Premium": 500, "VIP": 1500}
            profile.payment_amount = tier_prices.get(profile.membership_tier, 500)
        
        # Save to database
        doc = profile.model_dump()
        doc['created_at'] = doc['created_at'].isoformat()
        doc['updated_at'] = doc['updated_at'].isoformat()
        if doc.get('tier_expires_at'):
            doc['tier_expires_at'] = doc['tier_expires_at'].isoformat()
        
        await db.doctors.insert_one(doc)
        logger.info(f"Doctor registered: {profile.email} - Tier: {profile.membership_tier} - Early Bird: {profile.is_early_bird}")
        
        return profile
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/doctors", response_model=List[DoctorProfile])
async def get_doctors(
    medical_stream: Optional[str] = None,
    specialty: Optional[str] = None,
    city: Optional[str] = None,
    status: str = "approved",
    skip: int = 0,
    limit: int = 20
):
    """Get list of doctors with filters"""
    try:
        query = {"status": status}
        
        if medical_stream:
            query["medical_stream"] = medical_stream
        if specialty:
            query["specialty"] = specialty
        if city:
            query["city"] = city
        
        doctors = await db.doctors.find(query, {"_id": 0}).skip(skip).limit(limit).to_list(limit)
        
        # Convert ISO strings back to datetime
        for doctor in doctors:
            if isinstance(doctor.get('created_at'), str):
                doctor['created_at'] = datetime.fromisoformat(doctor['created_at'])
            if isinstance(doctor.get('updated_at'), str):
                doctor['updated_at'] = datetime.fromisoformat(doctor['updated_at'])
        
        return doctors
    except Exception as e:
        logger.error(f"Get doctors error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/doctors/{doctor_id}", response_model=DoctorProfile)
async def get_doctor_by_id(doctor_id: str):
    """Get doctor profile by ID"""
    try:
        doctor = await db.doctors.find_one({"id": doctor_id}, {"_id": 0})
        
        if not doctor:
            raise HTTPException(status_code=404, detail="Doctor not found")
        
        # Convert ISO strings back to datetime
        if isinstance(doctor.get('created_at'), str):
            doctor['created_at'] = datetime.fromisoformat(doctor['created_at'])
        if isinstance(doctor.get('updated_at'), str):
            doctor['updated_at'] = datetime.fromisoformat(doctor['updated_at'])
        
        # Increment profile views
        await db.doctors.update_one(
            {"id": doctor_id},
            {"$inc": {"profile_views": 1}, "$set": {"last_viewed_at": datetime.now(timezone.utc).isoformat()}}
        )
        
        return doctor
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get doctor error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.put("/doctors/{doctor_id}", response_model=DoctorProfile)
async def update_doctor_profile(doctor_id: str, updates: Dict[str, Any]):
    """Update doctor profile"""
    try:
        updates['updated_at'] = datetime.now(timezone.utc).isoformat()
        
        result = await db.doctors.update_one(
            {"id": doctor_id},
            {"$set": updates}
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Doctor not found")
        
        updated_doctor = await db.doctors.find_one({"id": doctor_id}, {"_id": 0})
        
        # Convert ISO strings back to datetime
        if isinstance(updated_doctor.get('created_at'), str):
            updated_doctor['created_at'] = datetime.fromisoformat(updated_doctor['created_at'])
        if isinstance(updated_doctor.get('updated_at'), str):
            updated_doctor['updated_at'] = datetime.fromisoformat(updated_doctor['updated_at'])
        
        return updated_doctor
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update doctor error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================
# ADMIN ENDPOINTS
# =====================================================

@api_router.get("/admin/doctors/pending", response_model=List[DoctorProfile])
async def get_pending_doctors(skip: int = 0, limit: int = 50):
    """Get pending doctor registrations for admin approval"""
    try:
        doctors = await db.doctors.find({"status": "pending"}, {"_id": 0}).skip(skip).limit(limit).to_list(limit)
        
        for doctor in doctors:
            if isinstance(doctor.get('created_at'), str):
                doctor['created_at'] = datetime.fromisoformat(doctor['created_at'])
            if isinstance(doctor.get('updated_at'), str):
                doctor['updated_at'] = datetime.fromisoformat(doctor['updated_at'])
        
        return doctors
    except Exception as e:
        logger.error(f"Get pending doctors error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/admin/doctors/{doctor_id}/approve")
async def approve_doctor(doctor_id: str, admin_id: str):
    """Approve doctor registration"""
    try:
        result = await db.doctors.update_one(
            {"id": doctor_id},
            {
                "$set": {
                    "status": "approved",
                    "approved_by": admin_id,
                    "approved_at": datetime.now(timezone.utc).isoformat(),
                    "updated_at": datetime.now(timezone.utc).isoformat()
                }
            }
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Doctor not found")
        
        return {"message": "Doctor approved successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Approve doctor error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/admin/doctors/{doctor_id}/reject")
async def reject_doctor(doctor_id: str, reason: str):
    """Reject doctor registration"""
    try:
        result = await db.doctors.update_one(
            {"id": doctor_id},
            {
                "$set": {
                    "status": "rejected",
                    "rejection_reason": reason,
                    "updated_at": datetime.now(timezone.utc).isoformat()
                }
            }
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Doctor not found")
        
        return {"message": "Doctor registration rejected"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Reject doctor error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================
# SEARCH & STATISTICS ENDPOINTS
# =====================================================

@api_router.post("/doctors/search", response_model=List[DoctorProfile])
async def search_doctors(filters: DoctorSearchFilters):
    """Advanced search for doctors"""
    try:
        query = {"status": "approved"}
        
        if filters.medical_stream:
            query["medical_stream"] = filters.medical_stream
        if filters.specialty:
            query["specialty"] = {"$regex": filters.specialty, "$options": "i"}
        if filters.city:
            query["city"] = {"$regex": filters.city, "$options": "i"}
        if filters.state:
            query["state"] = {"$regex": filters.state, "$options": "i"}
        if filters.gender:
            query["gender"] = filters.gender
        if filters.keyword:
            query["$or"] = [
                {"first_name": {"$regex": filters.keyword, "$options": "i"}},
                {"surname": {"$regex": filters.keyword, "$options": "i"}},
                {"specialty": {"$regex": filters.keyword, "$options": "i"}},
                {"areas_of_interest": {"$regex": filters.keyword, "$options": "i"}}
            ]
        
        doctors = await db.doctors.find(query, {"_id": 0}).skip(filters.skip).limit(filters.limit).to_list(filters.limit)
        
        for doctor in doctors:
            if isinstance(doctor.get('created_at'), str):
                doctor['created_at'] = datetime.fromisoformat(doctor['created_at'])
            if isinstance(doctor.get('updated_at'), str):
                doctor['updated_at'] = datetime.fromisoformat(doctor['updated_at'])
        
        return doctors
    except Exception as e:
        logger.error(f"Search doctors error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/statistics")
async def get_statistics():
    """Get platform statistics"""
    try:
        total_doctors = await db.doctors.count_documents({"status": "approved"})
        pending_doctors = await db.doctors.count_documents({"status": "pending"})
        
        # Early bird status
        early_bird_remaining = max(0, 100 - total_doctors)
        is_early_bird_available = total_doctors < 100
        
        # Count by stream
        streams = {}
        for stream in ["MBBS", "Dental", "Ayurveda", "Homeo", "Veterinary", "Pharma", "Physio"]:
            count = await db.doctors.count_documents({"medical_stream": stream, "status": "approved"})
            streams[stream] = count
        
        # Count by tier
        tier_counts = {}
        for tier in ["Essential", "Premium", "VIP"]:
            count = await db.doctors.count_documents({"membership_tier": tier, "status": "approved"})
            tier_counts[tier] = count
        
        # Count by location
        cities_pipeline = [
            {"$match": {"status": "approved"}},
            {"$group": {"_id": "$city", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 10}
        ]
        top_cities = await db.doctors.aggregate(cities_pipeline).to_list(10)
        
        return {
            "total_doctors": total_doctors,
            "pending_approvals": pending_doctors,
            "early_bird_remaining": early_bird_remaining,
            "is_early_bird_available": is_early_bird_available,
            "streams": streams,
            "membership_tiers": tier_counts,
            "top_cities": [{"city": c["_id"], "count": c["count"]} for c in top_cities]
        }
    except Exception as e:
        logger.error(f"Get statistics error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Include the router in the main app
app.include_router(api_router)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
