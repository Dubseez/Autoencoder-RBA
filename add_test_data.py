from app import db, LoginAttempts, app
from datetime import datetime

# Use the application context to access the database
with app.app_context():
    # Create a test login attempt
    test_attempt = LoginAttempts(
        user_id="test_user",
        ip_address="192.168.1.1",
        latitude=37.7749,
        longitude=-122.4194,
        timezone="PST",
        device_info="Windows 10, Chrome Browser",
        typing_speed=80.5,
        mouse_speed=120.3,
        login_time=datetime.utcnow()
    )

    # Add and commit the entry
    db.session.add(test_attempt)
    db.session.commit()

    print("Test login attempt added successfully!")
