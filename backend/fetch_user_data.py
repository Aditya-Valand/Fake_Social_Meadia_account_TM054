import instaloader

def fetch_user_data(username):

    # Create Instaloader session
    L = instaloader.Instaloader()


    try:
        # Retrieve profile metadata
        profile = instaloader.Profile.from_username(L.context, username)

        # Extract user data
        # user_data = {
        #     'username': profile.username,
        #     'followers': profile.followers,
        #     'following': profile.followees,
        #     'bio': profile.biography,
        #     'external_link': 1 if profile.external_url else 0,
        #     'profile_pic': 1 if profile.profile_pic_url else 0
        # }
        #   Extract user data
        user_data = {
            'username': profile.username,
            'username_length': len(profile.username),
            'verified': 1 if profile.is_verified else 0,
            'is_business_account': 1 if profile.is_business_account else 0,
            'external_link': 1 if profile.external_url else 0,
            'profile_pic': 1 if profile.profile_pic_url else 0,
            'followers': profile.followers,
            'following': profile.followees,
            'bio': profile.biography,
            'num_posts': profile.mediacount,
            }
        return user_data

    except instaloader.exceptions.ProfileNotExistsException:
        print(f'Profile "{username}" does not exist.')
        return None

# Example usage:
username = 'aditya_a4555'  # Replace with the desired Instagram username
user_data = fetch_user_data(username)
# if user_data:
#     print('Username:', user_data['username'])
#     print('Followers:', user_data['followers'])
#     print('Following:', user_data['following'])
#     print('Bio:', user_data['bio'])
#     print('External link:', 'Yes' if user_data['external_link'] else 'No')
#     print('Profile pic:', user_data['profile_pic'])
if user_data:
    print('Username:', user_data['username'])
    print('Username Length:', user_data['username_length'])
    print('Verified:', 'Yes' if user_data['verified'] else 'No')
    print('Business Account:', 'Yes' if user_data['is_business_account'] else 'No')
    print('External Link:', 'Yes' if user_data['external_link'] else 'No')
    print('Profile Pic:', 'Yes' if user_data['profile_pic'] else 'No')
    print('Followers:', user_data['followers'])
    print('Following:', user_data['following'])
    print('Bio:', user_data['bio'])
    print('Number of Posts:', user_data['num_posts'])