#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Optional, Any, Union, List, Tuple
import sys
import os
import time
import requests
import re
import numpy as np
from datetime import datetime, timedelta
import anthropic
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.common.keys import Keys

from utils.logger import logger
from utils.browser import browser
from config import config

class ETHBTCCorrelationBot:
    def __init__(self) -> None:
        self.browser = browser
        self.config = config
        self.session = requests.Session()
        self.claude_client = anthropic.Client(api_key=self.config.CLAUDE_API_KEY)
        self.session.timeout = (30, 90)  # (connect, read) timeouts
        self.past_predictions = []  # Store past predictions for spicy callbacks
        self.meme_phrases = {  # Add common crypto memes for analysis
            'pump': ['to the moon', 'number go up', 'wagmi'],
            'dump': ['buying the dip', 'this is fine', 'capitulation'],
            'sideways': ['crab market', 'wake me up when volatility returns', 'boring']
        }
        self.CORRELATION_THRESHOLD = 0.75  # High correlation threshold
        self.VOLUME_THRESHOLD = 0.60  # Volume correlation threshold
        self.TIME_WINDOW = 24  # Hours to analyze
        logger.log_startup()

    def start(self) -> None:
        """Main bot execution loop"""
        try:
            retry_count = 0
            max_setup_retries = 3
            
            while retry_count < max_setup_retries:
                if not self.browser.initialize_driver():
                    retry_count += 1
                    logger.logger.warning(f"Browser initialization attempt {retry_count} failed, retrying...")
                    time.sleep(10)
                    continue
                    
                if not self._login_to_twitter():
                    retry_count += 1
                    logger.logger.warning(f"Twitter login attempt {retry_count} failed, retrying...")
                    time.sleep(15)
                    continue
                    
                break
            
            if retry_count >= max_setup_retries:
                raise Exception("Failed to initialize bot after maximum retries")

            logger.logger.info("Bot initialized successfully")

            while True:
                try:
                    self._run_correlation_cycle()
                    time.sleep(60)  # Run every minute for testing
                except Exception as e:
                    logger.log_error("Correlation Cycle", str(e), exc_info=True)
                    time.sleep(5 * 60)
                    continue

        except KeyboardInterrupt:
            logger.logger.info("Bot stopped by user")
        except Exception as e:
            logger.log_error("Bot Execution", str(e))
        finally:
            self._cleanup()

    def _get_crypto_data(self) -> Optional[Dict[str, Any]]:
        """Fetch BTC, ETH and top 10 crypto data from CoinGecko with retries"""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                response = self.session.get(
                    self.config.get_coingecko_markets_url(),
                    params={
                        **self.config.get_coingecko_params(),
                        'per_page': 10,  # Get top 10 coins
                        'order': 'market_cap_desc',
                        'sparkline': True  # Get historical data``
                    },
                    timeout=(30, 90)
                )
                response.raise_for_status()
                logger.log_coingecko_request("/markets", success=True)
                
                data = {
                    coin['symbol'].upper(): {
                        'current_price': coin['current_price'],
                        'volume': coin['total_volume'],
                        'price_change_percentage_24h': coin['price_change_percentage_24h'],
                        'sparkline': coin.get('sparkline_in_7d', {}).get('price', []),
                        'market_cap': coin['market_cap']
                    } for coin in response.json()
                }
                
                # Ensure we have at least BTC and ETH
                if 'BTC' not in data or 'ETH' not in data:
                    logger.log_error("Crypto Data", "Missing BTC or ETH data")
                    return None
                
                logger.logger.info(f"Successfully fetched crypto data for {', '.join(data.keys())}")
                return data
                
            except requests.exceptions.Timeout:
                retry_count += 1
                wait_time = retry_count * 10
                logger.logger.warning(f"CoinGecko timeout, attempt {retry_count}, waiting {wait_time}s...")
                time.sleep(wait_time)
                
            except Exception as e:
                logger.log_coingecko_request("/markets", success=False)
                logger.log_error("CoinGecko API", str(e))
                return None
        
        logger.log_error("CoinGecko API", "Maximum retries reached")
        return None

    def _calculate_correlations(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate price and volume correlations"""
        coins = list(market_data.keys())
        n_coins = len(coins)
        
        # Create correlation matrices
        price_matrix = np.zeros((n_coins, n_coins))
        volume_matrix = np.zeros((n_coins, n_coins))
        
        # Calculate correlations
        for i, coin1 in enumerate(coins):
            for j, coin2 in enumerate(coins):
                if i != j:
                    # Price correlation
                    price_change1 = market_data[coin1]['price_change_percentage_24h']
                    price_change2 = market_data[coin2]['price_change_percentage_24h']
                    price_matrix[i][j] = abs(price_change1 - price_change2) < 1.0
                    
                    # Volume correlation
                    vol1 = market_data[coin1]['volume']
                    vol2 = market_data[coin2]['volume']
                    volume_matrix[i][j] = abs((vol1 - vol2) / max(vol1, vol2)) < 0.2
        
        # Find significant correlations
        correlations = {
            'price_pairs': [],
            'volume_pairs': [],
            'inverse_pairs': []
        }
        
        for i, coin1 in enumerate(coins):
            for j, coin2 in enumerate(coins[i+1:], i+1):
                # Check price correlation
                if price_matrix[i][j] > self.CORRELATION_THRESHOLD:
                    correlations['price_pairs'].append((coin1, coin2))
                elif price_matrix[i][j] < -self.CORRELATION_THRESHOLD:
                    correlations['inverse_pairs'].append((coin1, coin2))
                    
                # Check volume correlation
                if volume_matrix[i][j] > self.VOLUME_THRESHOLD:
                    correlations['volume_pairs'].append((coin1, coin2))
        
        return correlations

    def _track_prediction(self, prediction: Dict[str, Any], relevant_coins: List[str]) -> None:
        """Track predictions for future spicy callbacks"""
        MAX_PREDICTIONS = 20  # Keep more for better callback opportunities
        current_prices = {coin: prediction[f'{coin.lower()}_price'] for coin in relevant_coins}
        
        self.past_predictions.append({
            'timestamp': datetime.now(),
            'prediction': prediction['analysis'],
            'prices': current_prices,
            'sentiment': prediction['sentiment'],
            'outcome': None
        })
        
        # Cleanup old predictions
        self.past_predictions = [p for p in self.past_predictions 
                               if (datetime.now() - p['timestamp']).total_seconds() < 86400]  # 24 hours

    def _validate_past_prediction(self, prediction: Dict[str, Any], current_prices: Dict[str, float]) -> str:
        """Check if a past prediction was hilariously wrong"""
        sentiment_map = {
            'bullish': 1,
            'bearish': -1,
            'neutral': 0
        }
        
        # Calculate if prediction was right
        was_right = True
        for coin, old_price in prediction['prices'].items():
            if coin in current_prices:
                price_change = ((current_prices[coin] - old_price) / old_price) * 100
                sentiment_direction = sentiment_map.get(prediction['sentiment'], 0)
                
                # If sentiment direction doesn't match price movement by at least 2%
                if (sentiment_direction * price_change) < -2:
                    was_right = False
                    break
        
        return 'wrong' if not was_right else 'right'

    def _get_spicy_callback(self, current_prices: Dict[str, float]) -> Optional[str]:
        """Generate witty callbacks to past terrible predictions"""
        recent_predictions = [p for p in self.past_predictions 
                            if p['timestamp'] > (datetime.now() - timedelta(hours=24))]
        
        if not recent_predictions:
            return None
            
        # Update outcomes for past predictions
        for pred in recent_predictions:
            if pred['outcome'] is None:
                pred['outcome'] = self._validate_past_prediction(pred, current_prices)
                
        # Find wrong predictions to make fun of
        wrong_predictions = [p for p in recent_predictions if p['outcome'] == 'wrong']
        if wrong_predictions:
            worst_pred = wrong_predictions[-1]
            time_ago = int((datetime.now() - worst_pred['timestamp']).total_seconds() / 3600)
            
            # Generate spicy callback
            callbacks = [
                f"(Remember when I said {worst_pred['prediction']} {time_ago}h ago? Yeah... let's pretend that never happened)",
                f"(Unlike my galaxy-brain take {time_ago}h ago about {worst_pred['prediction']}... this time I'm sure!)",
                f"(Looks like my {time_ago}h old prediction of {worst_pred['prediction']} aged like milk. But trust me bro!)"
            ]
            return callbacks[hash(str(datetime.now())) % len(callbacks)]
            
        return None

    def _analyze_market_sentiment(self, crypto_data: Dict[str, Any]) -> Optional[str]:
        """Generate spicy market analysis with enhanced pattern detection"""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Calculate correlations
                correlations = self._calculate_correlations(crypto_data)
                
                # Get spicy callback about past predictions
                callback = self._get_spicy_callback({sym: data['current_price'] 
                                                   for sym, data in crypto_data.items()})
                
                # Prepare market mood
                avg_change = sum(data['price_change_percentage_24h'] for data in crypto_data.values()) / len(crypto_data)
                market_mood = 'bearish' if avg_change < 0 else 'bullish'
                mood_phrases = self.meme_phrases['dump'] if market_mood == 'bearish' else self.meme_phrases['pump']
                
                prompt = f"""Write a witty crypto market analysis as a single paragraph. Market data:
                - Top 10 Coins: {', '.join([f"{k} ({v['price_change_percentage_24h']:.1f}%)" for k, v in crypto_data.items()])}
                - Correlated Pairs: {correlations['price_pairs']}
                - Inverse Pairs: {correlations['inverse_pairs']}
                - Volume Correlations: {correlations['volume_pairs']}
                - Market Mood: {market_mood}
                - Common Phrases: {mood_phrases}
                - Past Context: {callback if callback else 'None'}

                Requirements:
                1. Be bold and sarcastic about market psychology
                2. Reference specific correlations or patterns
                3. Include crypto memes/jokes
                4. If applicable, include self-deprecating callback
                5. Keep it market-relevant but spicy
                6. Write as ONE cohesive paragraph"""
                
                response = self.claude_client.messages.create(
                    model=self.config.CLAUDE_MODEL,
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                analysis = response.content[0].text
                
                # Track this prediction
                prediction_data = {
                    'analysis': analysis,
                    'sentiment': market_mood,
                    **{f"{sym.lower()}_price": data['current_price'] for sym, data in crypto_data.items()}
                }
                self._track_prediction(prediction_data, list(crypto_data.keys()))
                
                return self._format_tweet_analysis(analysis, crypto_data)
                
            except Exception as e:
                retry_count += 1
                wait_time = retry_count * 10
                logger.logger.warning(f"Analysis error, attempt {retry_count}, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
        
        logger.log_error("Market Analysis", "Maximum retries reached")
        return None

    def _format_tweet_analysis(self, analysis: str, crypto_data: Dict[str, Any]) -> str:
        """Format analysis for Twitter"""
        # Only include analysis and hashtags
        tweet = f"{analysis}\n\n#Crypto #Trading"
        
        # Ensure we don't exceed tweet length
        max_length = self.config.TWEET_CONSTRAINTS['HARD_STOP_LENGTH'] - 20
        if len(tweet) > max_length:
            analysis = analysis[:max_length-23] + "..."  # 23 accounts for hashtags and ellipsis
            tweet = f"{analysis}\n\n#Crypto #Trading"
        
        return tweet

    def _run_correlation_cycle(self) -> None:
        """Run correlation analysis and posting cycle"""
        try:
            market_data = self._get_crypto_data()
            if not market_data:
                logger.logger.error("Failed to fetch market data")
                return
                
            analysis = self._analyze_market_sentiment(market_data)
            if not analysis:
                logger.logger.error("Failed to generate analysis")
                return
                
            # Check for duplicates
            last_posts = self._get_last_posts()
            if not self._is_duplicate_analysis(analysis, last_posts):
                if self._post_analysis(analysis):
                    logger.logger.info("Successfully posted analysis")
                else:
                    logger.logger.error("Failed to post analysis")
            else:
                logger.logger.info("Skipping duplicate analysis")
                
        except Exception as e:
            logger.log_error("Analysis Cycle", str(e))

    def _get_last_posts(self) -> List[str]:
        """Get last 10 posts to check for duplicates"""
        try:
            self.browser.driver.get(f'https://twitter.com/{self.config.TWITTER_USERNAME}')
            time.sleep(3)
            
            posts = WebDriverWait(self.browser.driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, '[data-testid="tweetText"]'))
            )
            
            return [post.text for post in posts[:10]]
        except Exception as e:
            logger.log_error("Get Last Posts", str(e))
            return []

    def _is_duplicate_analysis(self, new_tweet: str, last_posts: List[str]) -> bool:
        """Check if analysis is a duplicate"""
        try:
            for post in last_posts:
                if post.strip() == new_tweet.strip():
                    return True
            return False
        except Exception as e:
            logger.log_error("Duplicate Check", str(e))
            return False

    def _login_to_twitter(self) -> bool:
        """Log into Twitter with enhanced verification"""
        try:
            logger.logger.info("Starting Twitter login")
            self.browser.driver.set_page_load_timeout(45)
            self.browser.driver.get('https://twitter.com/login')
            time.sleep(5)

            # Username entry
            username_field = WebDriverWait(self.browser.driver, 20).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "input[autocomplete='username']"))
            )
            username_field.click()
            time.sleep(1)
            username_field.send_keys(self.config.TWITTER_USERNAME)
            time.sleep(2)

            # Click next
            next_button = WebDriverWait(self.browser.driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//span[text()='Next']"))
            )
            next_button.click()
            time.sleep(3)

            # Password entry
            password_field = WebDriverWait(self.browser.driver, 20).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "input[type='password']"))
            )
            password_field.click()
            time.sleep(1)
            password_field.send_keys(self.config.TWITTER_PASSWORD)
            time.sleep(2)

            # Login
            login_button = WebDriverWait(self.browser.driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//span[text()='Log in']"))
            )
            login_button.click()
            time.sleep(10)  # Wait for 2FA

            return self._verify_login()

        except Exception as e:
            logger.log_error("Twitter Login", str(e))
            return False

    def _verify_login(self) -> bool:
        """Verify Twitter login success"""
        try:
            verification_methods = [
                lambda: WebDriverWait(self.browser.driver, 30).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, '[data-testid="SideNav_NewTweet_Button"]'))
                ),
                lambda: WebDriverWait(self.browser.driver, 30).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, '[data-testid="AppTabBar_Profile_Link"]'))
                ),
                lambda: any(path in self.browser.driver.current_url 
                          for path in ['home', 'twitter.com/home'])
            ]
            
            for method in verification_methods:
                try:
                    if method():
                        return True
                except:
                    continue
            
            return False
            
        except Exception as e:
            logger.log_error("Login Verification", str(e))
            return False

    def _post_analysis(self, tweet_text: str) -> bool:
        """Post analysis to Twitter with robust button handling"""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                self.browser.driver.get('https://twitter.com/compose/tweet')
                time.sleep(3)
                
                text_area = WebDriverWait(self.browser.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, '[data-testid="tweetTextarea_0"]'))
                )
                text_area.click()
                time.sleep(1)
                
                # Send text in chunks
                text_parts = tweet_text.split('#')
                text_area.send_keys(text_parts[0])
                time.sleep(1)
                for part in text_parts[1:]:
                    text_area.send_keys(f'#{part}')
                    time.sleep(0.5)
                
                time.sleep(2)

                # Try multiple methods to click post button
                post_button = None
                button_locators = [
                    (By.CSS_SELECTOR, '[data-testid="tweetButton"]'),
                    (By.XPATH, "//div[@role='button'][contains(., 'Post')]"),
                    (By.XPATH, "//span[text()='Post']")
                ]

                for locator in button_locators:
                    try:
                        post_button = WebDriverWait(self.browser.driver, 5).until(
                            EC.element_to_be_clickable(locator)
                        )
                        if post_button:
                            break
                    except:
                        continue

                if post_button:
                    self.browser.driver.execute_script("arguments[0].scrollIntoView(true);", post_button)
                    time.sleep(1)
                    self.browser.driver.execute_script("arguments[0].click();", post_button)
                    time.sleep(5)
                    logger.logger.info("Tweet posted successfully")
                    return True
                else:
                    logger.logger.error("Could not find post button")
                    retry_count += 1
                    time.sleep(2)
                    
            except Exception as e:
                logger.logger.error(f"Tweet posting error, attempt {retry_count + 1}: {str(e)}")
                retry_count += 1
                wait_time = retry_count * 10
                logger.logger.warning(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                continue
        
        logger.log_error("Tweet Creation", "Maximum retries reached")
        return False

    def _cleanup(self) -> None:
        """Cleanup resources"""
        try:
            if self.browser:
                logger.logger.info("Closing browser...")
                try:
                    self.browser.close_browser()
                    time.sleep(1)
                except Exception as e:
                    logger.logger.warning(f"Error during browser close: {str(e)}")
            logger.log_shutdown()
        except Exception as e:
            logger.log_error("Cleanup", str(e))

if __name__ == "__main__":
    bot = ETHBTCCorrelationBot()
    bot.start()                
