import openai
import streamlit as st
from openai import OpenAI
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from difflib import SequenceMatcher

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

def initialize_session_state():
    """Initialize session state variables."""
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'openai_client' not in st.session_state:
        # Set up OpenAI client
        api_key = st.secrets["openai"]["api_key"]  # Store your API key in streamlit secrets
        st.session_state.openai_client = OpenAI(api_key=api_key)

def generate_dynamic_prompt(history):
    """Generate a dynamic system prompt based on the conversation history."""
    base_prompt = """You are an empathetic AI companion focused on supportive listening and emotional well-being. Your approach is:

    Core Qualities:
    - Deeply empathetic and non-judgmental
    - Patient and attentive to emotional nuances
    - Gentle but clear in communication
    - Professional while maintaining warmth
    
    Interaction Guidelines:
    - Practice active listening and reflection
    - Validate emotions without immediately trying to fix them
    - Ask open-ended questions to explore feelings
    - Share coping strategies when appropriate
    - Use "I hear you" and similar validating phrases
    - Maintain appropriate boundaries
    
    Safety Protocol:
    - For any mentions of self-harm, suicide, or severe crisis, ALWAYS respond with:
      "I hear that you're going through something very difficult. However, I'm an AI and not qualified to help with crisis situations. Please reach out to professional help immediately:
      - Crisis Helpline: 988 (US)
      - Emergency Services: 911 (US)
      - Or contact a mental health professional directly"
    
    Conversation Approach:
    - Start with understanding before suggesting
    - Break down overwhelming feelings into manageable parts
    - Encourage self-reflection
    - Suggest mindfulness and grounding techniques when relevant
    - Celebrate small progress and victories
    
    Remember: While you can provide emotional support and a listening ear, always be clear that you are an AI companion, not a replacement for professional mental health care.
    """
    
    # Extract user messages from history
    user_messages = [msg["content"] for msg in history if msg["role"] == "user"]
    last_3_user = user_messages[-3:] if len(user_messages) >= 3 else user_messages
    
    if not last_3_user:
        return base_prompt, {}, {}  # Return empty metrics if no messages
    
    # Analyze tone using VADER sentiment analysis
    sentiment_scores = [analyzer.polarity_scores(msg) for msg in last_3_user]
    avg_compound = sum(s["compound"] for s in sentiment_scores) / len(sentiment_scores)
    
    # More nuanced tone analysis
    if avg_compound < -0.5:
        tone = "distressed"
    elif avg_compound < -0.2:
        tone = "concerned"
    elif avg_compound < 0.1:
        tone = "neutral"
    elif avg_compound < 0.5:
        tone = "positive"
    else:
        tone = "very_positive"
    
    # Analyze engagement based on average word count
    word_counts = [len(msg.split()) for msg in last_3_user]
    avg_word_count = sum(word_counts) / len(word_counts)
    engagement = "disengaged" if avg_word_count < 3 else "engaged"
    
    # Build dynamic prompt with more specific guidance
    dynamic_prompt = base_prompt
    if tone == "distressed":
        dynamic_prompt += """
        The user appears to be in significant distress. Priority response guidelines:
        - Acknowledge their pain immediately
        - Use more validation statements
        - Keep responses gentle and supportive
        - Focus on immediate emotional support
        - Be extra careful with suggestions or advice
        - Check if they have support systems in place
        """
    elif tone == "concerned":
        dynamic_prompt += """
        The user seems worried or troubled. Approach guidelines:
        - Validate their concerns
        - Use gentle exploration questions
        - Offer grounding techniques if appropriate
        - Help identify specific worries
        """
    
    # Prepare metrics for display
    sentiment_metrics = {
        "Compound Score": round(avg_compound, 3),
        "Overall Tone": tone.capitalize()
    }
    
    engagement_metrics = {
        "Avg Word Count": round(avg_word_count, 1),
        "Engagement Level": engagement.capitalize()
    }
    
    return dynamic_prompt, sentiment_metrics, engagement_metrics

def check_response_repetition(history, new_response):
    """Check if AI responses are becoming repetitive without progression"""
    # Get last 4 AI messages
    ai_messages = [msg["content"] for msg in history[-8:] if msg["role"] == "assistant"][-4:]
    if len(ai_messages) < 2:  # Need at least 2 messages to check repetition
        return False, ""
    
    # Calculate similarity ratios between consecutive messages
    similarities = []
    for i in range(len(ai_messages)-1):
        ratio = SequenceMatcher(None, ai_messages[i], ai_messages[i+1]).ratio()
        similarities.append(ratio)
    
    # Calculate similarity of new response with last response
    new_similarity = SequenceMatcher(None, ai_messages[-1], new_response).ratio()
    similarities.append(new_similarity)
    
    # Check if recent responses are too similar
    avg_similarity = sum(similarities) / len(similarities)
    is_repetitive = avg_similarity > 0.7  # Threshold for similarity
    
    adjustment = """
    The conversation needs gentle redirection. Consider:
    1. Exploring a different aspect of their feelings
    2. Asking about specific examples or situations
    3. Introducing a gentle coping technique
    4. Checking in about their current emotional state
    5. Reflecting on any changes they've noticed
    
    Maintain the supportive tone while gently moving the conversation forward.
    """
    
    return is_repetitive, adjustment

def main():
    st.title("AI Emotional Support Companion")
    st.markdown("""
    > **Important**: This AI companion provides emotional support but is NOT a replacement for professional mental health care. 
    > If you're in crisis, please call 988 (US) or your local emergency services immediately.
    """)
    initialize_session_state()
    
    # Create persistent containers for metrics
    if 'metrics_container' not in st.session_state:
        st.session_state.metrics_container = st.sidebar.empty()
    if 'update_counter' not in st.session_state:
        st.session_state.update_counter = 0
    
    def update_metrics():
        """Update the metrics display in the sidebar"""
        # Increment counter for unique keys
        st.session_state.update_counter += 1
        counter = st.session_state.update_counter
        
        with st.session_state.metrics_container.container():
            st.title("Conversation Analytics")
            st.markdown("### Current Dynamic Prompt")
            
            # Get the current prompt and metrics
            current_prompt, sentiment_metrics, engagement_metrics = generate_dynamic_prompt(st.session_state.history)
            
            # Display the prompt with unique key
            st.text_area(
                "System Prompt",
                current_prompt,
                height=150,
                disabled=True,
                key=f"system_prompt_display_{counter}"
            )
            
            # Display metrics with unique keys
            st.markdown("### Sentiment Analysis")
            for i, (key, value) in enumerate(sentiment_metrics.items()):
                st.text_input(
                    key,
                    value=str(value),
                    disabled=True,
                    key=f"sentiment_metric_{counter}_{i}"
                )
                
            st.markdown("### Engagement Metrics")
            for i, (key, value) in enumerate(engagement_metrics.items()):
                st.text_input(
                    key,
                    value=str(value),
                    disabled=True,
                    key=f"engagement_metric_{counter}_{i}"
                )
    
    # Initial metrics display
    update_metrics()
    
    # Display chat messages
    for message in st.session_state.history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if user_input := st.chat_input("Type your message here..."):
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)
        
        # Add user message to history
        st.session_state.history.append({"role": "user", "content": user_input})
        
        # Update metrics immediately
        update_metrics()
        
        # Generate dynamic prompt for AI
        dynamic_prompt, _, _ = generate_dynamic_prompt(st.session_state.history)
        
        try:
            # Prepare messages for OpenAI API
            messages = [{"role": "system", "content": dynamic_prompt}] + st.session_state.history
            
            while True:  # Loop to handle repetitive responses
                with st.spinner("Thinking..."):
                    response = st.session_state.openai_client.chat.completions.create(
                        model="gpt-4",
                        messages=messages,
                        temperature=0.8
                    )
                
                # Extract AI response
                ai_response = response.choices[0].message.content
                
                # Check for repetition
                is_repetitive, adjustment = check_response_repetition(st.session_state.history, ai_response)
                
                if not is_repetitive:
                    break
                
                # If repetitive, add adjustment to prompt and try again
                messages.append({"role": "system", "content": adjustment})
            
            # Add final response to history
            st.session_state.history.append({"role": "assistant", "content": ai_response})
            
            # Display AI response
            with st.chat_message("assistant"):
                st.write(ai_response)
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
            with st.chat_message("assistant"):
                st.write("Sorry, I'm having trouble right now. Please try again later.")

if __name__ == "__main__":
    main()