from datetime import datetime

def reformat_to_bio(json_data):
    # Extract _id
    _id = json_data.get("_id", "")

    # Calculate age from dateOfBirth
    dob = json_data.get("dateOfBirth", "")
    age = None
    if dob:
        try:
            birth_year = int(dob.split("-")[0])
            current_year = datetime.now().year  # 2025 based on system date
            age = current_year - birth_year
        except (ValueError, IndexError):
            age = None

    # Convert height from cm to feet/inches
    height_cm = json_data.get("height", None)
    height_str = ""
    if height_cm:
        try:
            height_cm = float(height_cm)
            total_inches = height_cm / 2.54  # Convert cm to inches
            feet = int(total_inches // 12)
            inches = int(total_inches % 12)
            height_str = f"{feet}'{inches}\""
        except (ValueError, TypeError):
            height_str = ""

    # Collect fields for bio
    first_name = json_data.get("firstName", "")
    gender = json_data.get("gender", "").capitalize()
    religion = json_data.get("religion", "").capitalize()
    education = json_data.get("education", "")
    hobbies = json_data.get("hobbies", [])
    interests = json_data.get("interests", [])
    spoken_languages = json_data.get("spokenLanguages", [])
    what_brings_you_here = json_data.get("whatBringsYouHere", "")
    min_age = json_data.get("minAge", "")
    max_age = json_data.get("maxAge", "")
    gender_interest = json_data.get("genderInterest", "").capitalize()
    looking_for = json_data.get("lookingFor", [])
    favorite_colors = json_data.get("favoriteColors", [])
    pets = json_data.get("pets", [])

    # Build bio components
    bio_parts = []

    # Intro: name, age, gender, height, religion, education
    intro_parts = []
    if first_name:
        intro_parts.append(f"I'm {first_name}")
    if age is not None:
        intro_parts.append(f"a {age}-year-old")
    if gender:
        intro_parts.append(gender.lower())
    if height_str:
        intro_parts.append(f"{height_str} tall")
    if religion:
        intro_parts.append(f"{religion}")
    if education:
        intro_parts.append(f"with a {education}")
    if intro_parts:
        bio_parts.append(", ".join(intro_parts))

    # Hobbies and interests
    activities = []
    if hobbies:
        activities.extend(hobbies)
    if interests:
        activities.extend(interests)
    if activities:
        bio_parts.append(f"who loves {', '.join(activities)}")

    # Favorite colors
    if favorite_colors:
        bio_parts.append(f"I love the colors {', '.join(favorite_colors)}")

    # Pets
    if pets:
        bio_parts.append(f"I have {', '.join(pets)}")

    # Spoken languages
    if spoken_languages:
        bio_parts.append(f"I speak {', '.join(spoken_languages)}")

    # Purpose, gender interest, looking for, and age preference
    if looking_for or what_brings_you_here:
        # Combine lookingFor and whatBringsYouHere
        purposes = set(looking_for) | {
            what_brings_you_here} if what_brings_you_here else set(looking_for)
        purposes = [p for p in purposes if p]  # Remove empty strings
        purpose_str = " or ".join(purposes) if purposes else "connections"
        purpose = f"I'm here looking for {gender_interest.lower()} {purpose_str}"
        if min_age and max_age:
            purpose += f" aged {min_age} to {max_age}"
        bio_parts.append(purpose)

    # Combine bio parts into a single string
    bio = ". ".join(part for part in bio_parts if part).rstrip(".") + "."

    # Return reformatted JSON
    return {
        "_id": _id,
        "bio": bio
    }
