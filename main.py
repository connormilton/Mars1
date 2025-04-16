"""
Dual-Pair LLM Forex Trading System
Main entry point and system coordinator with enhanced reasoning and feedback
"""

import os
import time
import logging
import json
from datetime import datetime, timezone
import pandas as pd
import pandas_ta as ta  # For technical indicators
from dotenv import load_dotenv
from tabulate import tabulate  # For better terminal output
import colorama  # For colored terminal output
from colorama import Fore, Style

from agents import AnalysisAgent, ExecutionAgent, ReviewAgent
from utils import OandaClient, calculate_position_size, format_trade_log

# Initialize colorama for colored terminal output
colorama.init()

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("trading.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ForexTrader")

# Trading configuration
CONFIG = {
    "pairs": ["EUR_USD", "EUR_GBP"],
    "primary_pair": "EUR_USD",
    "secondary_pair": "EUR_GBP",
    "timeframes": {
        "m5": {"granularity": "M5", "count": 100},   # 5-minute charts
        "m15": {"granularity": "M15", "count": 96},  # 24 hours
        "h1": {"granularity": "H1", "count": 48},    # 2 days
        "h4": {"granularity": "H4", "count": 30},    # 5 days
        "d1": {"granularity": "D", "count": 20},     # 20 days
        "w1": {"granularity": "W", "count": 10}      # 10 weeks
    },
    "indicators": {
        "sma": [20, 50, 200],      # Simple Moving Averages
        "ema": [8, 21, 55],        # Exponential Moving Averages
        "rsi": {"period": 14},     # Relative Strength Index
        "macd": {"fast": 12, "slow": 26, "signal": 9},  # MACD
        "bbands": {"period": 20, "std_dev": 2},       # Bollinger Bands
        "atr": {"period": 14}      # Average True Range
    },
    "cycle_minutes": 5,                     # Minutes between trading cycles
    "max_positions_per_pair": 2,            # Maximum positions per currency pair
    "max_total_positions": 3,               # Maximum total open positions
    "min_total_positions": 1,               # Minimum total open positions
    "base_risk_percent": 1.5,               # Base risk percentage (1-5%)
    "max_risk_per_pair": 6.0,               # Maximum risk per pair (%)
    "max_total_risk": 12.0,                 # Maximum total account risk (%)
    "min_quality_score": 7.0,               # Minimum analysis quality to execute trade
    "min_risk_reward": 1.5,                 # Minimum risk-reward ratio
    "breakeven_move_pips": {                # Pips move needed to move stop to breakeven
        "EUR_USD": 15,
        "EUR_GBP": 20
    },
    "default_stop_pips": {                  # Default stop distance if not specified
        "EUR_USD": 30,
        "EUR_GBP": 40
    },
    "max_daily_loss_pct": 3.0,              # Maximum daily loss percentage (stop trading)
    "partial_profit_pct": 50,               # Percentage to close at first take profit
    "models": {
        "analysis": "gpt-4-turbo-preview",  # Model for analysis agent (more powerful)
        "execution": "gpt-3.5-turbo",       # Model for execution agent (faster/cheaper)
        "review": "gpt-4-turbo-preview"     # Model for review agent (high quality feedback)
    },
    "daily_budget": float(os.getenv("DAILY_LLM_BUDGET", 20.0)),
    "llm_provider": os.getenv("LLM_PROVIDER", "OpenAI")
}

class TradingSystem:
    """Dual-Pair LLM Forex Trading System"""
    
    def __init__(self):
        """Initialize the trading system"""
        self.config = CONFIG
        self.oanda = OandaClient()
        self.trading_memory = self._initialize_memory()
        
        # Initialize LLM agents
        self.analysis_agent = AnalysisAgent(
            model=self.config["models"]["analysis"],
            provider=self.config["llm_provider"],
            budget_manager=self._track_usage
        )
        
        self.execution_agent = ExecutionAgent(
            model=self.config["models"]["execution"],
            provider=self.config["llm_provider"],
            budget_manager=self._track_usage
        )
        
        self.review_agent = ReviewAgent(
            model=self.config["models"]["review"],
            provider=self.config["llm_provider"],
            budget_manager=self._track_usage
        )
        
        # Initialize session budget
        self.session_usage = {
            "budget": self.config["daily_budget"],
            "spent": 0.0,
            "last_reset": datetime.now(timezone.utc).date().isoformat()
        }
        
        # Initialize market regime data
        self.market_regime = {pair: "unknown" for pair in self.config["pairs"]}
        
        # Initialize daily P&L tracking
        self.daily_pnl = {
            "date": datetime.now(timezone.utc).date().isoformat(),
            "start_balance": 0.0,
            "current_balance": 0.0,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "total_pnl_pct": 0.0
        }
        
    def _initialize_memory(self):
        """Initialize or load trading memory"""
        try:
            if os.path.exists("memory.json"):
                with open("memory.json", "r") as f:
                    memory = json.load(f)
                    logger.info(f"Loaded existing memory with {len(memory.get('trades', []))} trade records")
                    return memory
        except Exception as e:
            logger.error(f"Error loading memory: {e}")
            
        # Create new memory structure
        memory = {
            "trades": [],
            "analyses": {},
            "last_cycle": "",
            "performance": {
                "win_count": 0,
                "loss_count": 0,
                "total_return_pct": 0.0,
                "largest_win_pct": 0.0,
                "largest_loss_pct": 0.0
            },
            "feedback": {
                "analysis": "",
                "execution": "",
                "review": ""
            },
            "learning": {
                "successful_patterns": {},
                "failed_patterns": {},
                "timeframe_effectiveness": {},
                "indicator_reliability": {}
            },
            "human_recommendations": []
        }
        return memory
    
    def save_memory(self):
        """Save trading memory to disk"""
        try:
            with open("memory.json", "w") as f:
                json.dump(self.trading_memory, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving memory: {e}")
            
    def _track_usage(self, cost, agent_name):
        """Track LLM API usage"""
        # Reset budget if it's a new day
        today = datetime.now(timezone.utc).date().isoformat()
        if today != self.session_usage["last_reset"]:
            self.session_usage = {
                "budget": self.config["daily_budget"],
                "spent": 0.0,
                "last_reset": today
            }
            
        # Update spent amount
        self.session_usage["spent"] += cost
        logger.info(f"LLM usage: ${cost:.4f} for {agent_name}, total: ${self.session_usage['spent']:.2f} of ${self.session_usage['budget']:.2f}")
        
        # Return remaining budget
        return self.session_usage["budget"] - self.session_usage["spent"]
    
    def calculate_market_regime(self, market_data, pair):
        """Determine the current market regime (trending, ranging, volatile, breakout)"""
        try:
            # Get daily candles
            daily_candles = market_data[pair]["d1"]
            if not daily_candles or len(daily_candles) < 20:
                return "unknown"
                
            # Convert to pandas DataFrame
            candles_df = pd.DataFrame([
                {
                    "open": float(candle["mid"]["o"]) if "mid" in candle else candle["open"],
                    "high": float(candle["mid"]["h"]) if "mid" in candle else candle["high"],
                    "low": float(candle["mid"]["l"]) if "mid" in candle else candle["low"],
                    "close": float(candle["mid"]["c"]) if "mid" in candle else candle["close"],
                    "volume": int(candle["volume"]) if "volume" in candle else 0
                }
                for candle in daily_candles
            ])
            
            # Calculate indicators
            candles_df["atr"] = ta.atr(candles_df["high"], candles_df["low"], candles_df["close"], length=14)
            candles_df["ema20"] = ta.ema(candles_df["close"], length=20)
            candles_df["ema50"] = ta.ema(candles_df["close"], length=50)
            
            # Get the most recent values
            recent_atr = candles_df["atr"].iloc[-1]
            recent_close = candles_df["close"].iloc[-1]
            ema20 = candles_df["ema20"].iloc[-1]
            ema50 = candles_df["ema50"].iloc[-1]
            
            # Calculate average daily range (last 10 days)
            ranges = candles_df["high"].iloc[-10:] - candles_df["low"].iloc[-10:]
            avg_range = ranges.mean()
            
            # Calculate directional movement (last 10 days)
            price_change = abs(candles_df["close"].iloc[-1] - candles_df["close"].iloc[-10])
            
            # Determine regime
            if recent_atr > avg_range * 1.5:
                regime = "volatile"
            elif price_change < avg_range * 2:
                regime = "ranging"
            elif ema20 > ema50 and candles_df["close"].iloc[-5:].min() > ema20:
                regime = "strong_uptrend"
            elif ema20 < ema50 and candles_df["close"].iloc[-5:].max() < ema20:
                regime = "strong_downtrend"
            elif ema20 > ema50:
                regime = "uptrend"
            elif ema20 < ema50:
                regime = "downtrend"
            else:
                regime = "ranging"
                
            return regime
            
        except Exception as e:
            logger.error(f"Error calculating market regime: {e}")
            return "unknown"
    
    def get_market_data(self):
        """Collect market data for configured pairs"""
        market_data = {}
        
        for pair in self.config["pairs"]:
            pair_data = {}
            
            # Get candles for each timeframe
            for tf_name, tf_config in self.config["timeframes"].items():
                candles = self.oanda.get_candles(
                    instrument=pair,
                    granularity=tf_config["granularity"],
                    count=tf_config["count"]
                )
                
                # Format candles
                pair_data[tf_name] = candles
            
            # Get current price
            price = self.oanda.get_price(pair)
            pair_data["current"] = price
            
            market_data[pair] = pair_data
            
            # Calculate market regime
            self.market_regime[pair] = self.calculate_market_regime(market_data, pair)
            
        return market_data
    
    def get_positions(self):
        """Get current open positions"""
        return self.oanda.get_open_positions()
    
    def get_account_info(self):
        """Get account information"""
        return self.oanda.get_account()
    
    def update_daily_pnl(self, account_info, positions):
        """Update daily P&L tracking"""
        today = datetime.now(timezone.utc).date().isoformat()
        
        # If a new day, reset the tracking
        if today != self.daily_pnl["date"]:
            self.daily_pnl = {
                "date": today,
                "start_balance": float(account_info.get("balance", 0)),
                "current_balance": float(account_info.get("balance", 0)),
                "realized_pnl": 0.0,
                "unrealized_pnl": 0.0,
                "total_pnl_pct": 0.0
            }
            return
            
        # If first update of the day, set the start balance
        if self.daily_pnl["start_balance"] == 0:
            self.daily_pnl["start_balance"] = float(account_info.get("balance", 0))
            
        # Update current values
        current_balance = float(account_info.get("balance", 0))
        unrealized_pnl = sum([float(position.get("profit", 0)) for position in positions])
        
        self.daily_pnl["current_balance"] = current_balance
        self.daily_pnl["unrealized_pnl"] = unrealized_pnl
        
        # Calculate total P&L percentage
        if self.daily_pnl["start_balance"] > 0:
            total_change = (current_balance - self.daily_pnl["start_balance"]) + unrealized_pnl
            self.daily_pnl["total_pnl_pct"] = (total_change / self.daily_pnl["start_balance"]) * 100
    
    def log_trade(self, trade_data):
        """Log a trade to memory"""
        # Add timestamp
        trade_data["timestamp"] = datetime.now(timezone.utc).isoformat()
        
        # Add to trades list
        self.trading_memory["trades"].append(trade_data)
        
        # Update performance metrics if it's a closed trade
        if trade_data.get("action_type") == "CLOSE":
            outcome = trade_data.get("outcome", "")
            
            if "WIN" in outcome or "PROFIT" in outcome:
                self.trading_memory["performance"]["win_count"] += 1
                
                # Update largest win if applicable
                return_pct = trade_data.get("return_percent", 0.0)
                if return_pct > self.trading_memory["performance"]["largest_win_pct"]:
                    self.trading_memory["performance"]["largest_win_pct"] = return_pct
                    
                # Track successful patterns
                pattern = trade_data.get("pattern", "unknown")
                if pattern in self.trading_memory["learning"]["successful_patterns"]:
                    self.trading_memory["learning"]["successful_patterns"][pattern] += 1
                else:
                    self.trading_memory["learning"]["successful_patterns"][pattern] = 1
                    
            elif "LOSS" in outcome or "STOPPED" in outcome:
                self.trading_memory["performance"]["loss_count"] += 1
                
                # Update largest loss if applicable
                return_pct = abs(trade_data.get("return_percent", 0.0))
                if return_pct > self.trading_memory["performance"]["largest_loss_pct"]:
                    self.trading_memory["performance"]["largest_loss_pct"] = return_pct
                
                # Track failed patterns
                pattern = trade_data.get("pattern", "unknown")
                if pattern in self.trading_memory["learning"]["failed_patterns"]:
                    self.trading_memory["learning"]["failed_patterns"][pattern] += 1
                else:
                    self.trading_memory["learning"]["failed_patterns"][pattern] = 1
            
            # Update realized P&L for the day
            profit = trade_data.get("profit", 0)
            self.daily_pnl["realized_pnl"] += float(profit)
        
        # Save to disk
        self.save_memory()
        
        # Log trade
        self.print_trade_info(trade_data)
        
    def print_trade_info(self, trade_data):
        """Print formatted trade information to terminal"""
        action_type = trade_data.get("action_type", "UNKNOWN")
        
        if action_type == "OPEN":
            message = (
                f"\n{Fore.CYAN}➡️  NEW TRADE{Style.RESET_ALL}\n"
                f"  {Fore.YELLOW}{trade_data.get('direction')}{Style.RESET_ALL} {trade_data.get('epic')} @ {trade_data.get('entry_price')}\n"
                f"  Size: {trade_data.get('size')} | Risk: {trade_data.get('risk_percent')}%\n"
                f"  Stop Loss: {trade_data.get('stop_loss')}\n"
                f"  Pattern: {trade_data.get('pattern')}\n"
            )
            print(message)
            
        elif action_type == "CLOSE":
            outcome = trade_data.get("outcome", "UNKNOWN")
            color = Fore.GREEN if "WIN" in outcome or "PROFIT" in outcome else Fore.RED
            
            message = (
                f"\n{color}✓  CLOSED POSITION{Style.RESET_ALL}\n"
                f"  {trade_data.get('epic')} @ {trade_data.get('close_price')}\n"
                f"  Profit/Loss: {color}{trade_data.get('profit')}{Style.RESET_ALL}\n"
                f"  Outcome: {color}{outcome}{Style.RESET_ALL}\n"
            )
            print(message)
            
        elif action_type == "UPDATE_STOP":
            message = (
                f"\n{Fore.YELLOW}⚙️  UPDATED STOP{Style.RESET_ALL}\n"
                f"  {trade_data.get('epic')} to {trade_data.get('new_level')}\n"
                f"  Reason: {trade_data.get('reason')}\n"
            )
            print(message)
            
        else:
            logger.info(f"Trade logged: {format_trade_log(trade_data)}")
        
    def get_recent_trades(self, limit=10):
        """Get recent trades from memory"""
        trades = self.trading_memory.get("trades", [])
        
        # Sort by timestamp (newest first)
        trades.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return trades[:limit]
    
    def update_stop_loss(self, position_data, new_stop):
        """Update stop loss for a position"""
        epic = position_data.get("epic")
        deal_id = position_data.get("dealId", position_data.get("deal_id"))
        
        success, update_result = self.oanda.update_stop_loss(epic, new_stop)
        
        if success:
            logger.info(f"Updated stop loss for {epic} to {new_stop}")
            
            # Log the update
            self.log_trade({
                "action_type": "UPDATE_STOP",
                "epic": epic,
                "deal_id": deal_id,
                "new_level": new_stop,
                "reason": "Trailing stop update"
            })
            
        return success
    
    def calculate_default_stop_loss(self, epic, direction, entry_price):
        """Calculate a default stop loss if none provided"""
        # Get default stop distance in pips
        default_pips = self.config["default_stop_pips"].get(epic, 30)
        
        # Convert to price movement
        pip_size = 0.01 if "JPY" in epic else 0.0001
        price_distance = default_pips * pip_size
        
        # Calculate stop loss based on direction
        if direction == "BUY":
            stop_loss = entry_price - price_distance
        else:  # SELL
            stop_loss = entry_price + price_distance
            
        logger.warning(f"Generated default stop loss for {epic} at {stop_loss} ({default_pips} pips from entry)")
        return stop_loss
            
    def execute_trades(self, execution_result):
        """Execute trades and position actions"""
        if not execution_result:
            return False
            
        trades_executed = 0
        
        # Get current positions and account info
        positions = self.get_positions()
        account_info = self.get_account_info()
        
        # Check if we've hit maximum daily loss
        if self.daily_pnl["total_pnl_pct"] <= -self.config["max_daily_loss_pct"]:
            logger.warning(f"Maximum daily loss of {self.config['max_daily_loss_pct']}% reached. Stopping trading for today.")
            print(f"\n{Fore.RED}⚠️  MAXIMUM DAILY LOSS REACHED ({self.daily_pnl['total_pnl_pct']:.2f}%). NO NEW TRADES ALLOWED.{Style.RESET_ALL}\n")
            return False
        
        # Execute new trades
        for trade in execution_result.get("trade_actions", []):
            epic = trade.get("epic")
            direction = trade.get("direction")
            risk_percent = float(trade.get("risk_percent", self.config["base_risk_percent"]))
            
            # Lower risk percentages to avoid margin issues
            # Cap risk at 2% maximum regardless of what was recommended
            if risk_percent > 2.0:
                logger.info(f"Reducing risk for {epic} from {risk_percent}% to 2.0% to avoid margin issues")
                risk_percent = 2.0
                
            entry_price = trade.get("entry_price")
            stop_loss = trade.get("initial_stop_loss")
            take_profit = trade.get("take_profit_levels", [])[0] if trade.get("take_profit_levels") else None
            
            # Validate required fields
            if not epic or not direction:
                logger.warning(f"Skipping trade - Missing required fields (epic: {epic}, direction: {direction})")
                continue
                
            # Ensure entry price is valid
            try:
                entry_price = float(entry_price) if entry_price else None
            except (ValueError, TypeError):
                logger.warning(f"Skipping {epic} {direction} - Invalid entry price: {entry_price}")
                continue
                
            # If stop loss is missing or invalid, calculate a default stop loss
            try:
                stop_loss = float(stop_loss) if stop_loss else None
            except (ValueError, TypeError):
                stop_loss = None
                
            if stop_loss is None and entry_price is not None:
                stop_loss = self.calculate_default_stop_loss(epic, direction, entry_price)
                
            # Check if we already have max positions
            current_total = len(positions)
            pair_positions = [p for p in positions if p.get("epic") == epic]
            current_pair = len(pair_positions)
            
            if current_total >= self.config["max_total_positions"]:
                logger.info(f"Skipping {epic} {direction} - Maximum total positions reached ({current_total})")
                continue
                
            if current_pair >= self.config["max_positions_per_pair"]:
                logger.info(f"Skipping {epic} {direction} - Maximum positions for {epic} reached ({current_pair})")
                continue
                
            # Check risk limits
            pair_risk = sum([float(p.get("risk_percent", 1.5)) for p in pair_positions])
            if pair_risk + risk_percent > self.config["max_risk_per_pair"]:
                logger.info(f"Skipping {epic} {direction} - Would exceed max risk per pair ({pair_risk + risk_percent:.1f}% > {self.config['max_risk_per_pair']}%)")
                continue
                
            total_risk = sum([float(p.get("risk_percent", 1.5)) for p in positions])
            if total_risk + risk_percent > self.config["max_total_risk"]:
                logger.info(f"Skipping {epic} {direction} - Would exceed max total risk ({total_risk + risk_percent:.1f}% > {self.config['max_total_risk']}%)")
                continue
            
            # Calculate position size
            account_balance = float(account_info.get("balance", 1000))
            units = calculate_position_size(
                account_balance=account_balance,
                risk_percent=risk_percent,
                entry_price=entry_price,
                stop_loss=stop_loss,
                instrument=epic,
                account_currency=account_info.get("currency", "USD")
            )
            
            # Apply direction
            if direction == "SELL":
                units = -abs(units)
            else:
                units = abs(units)
                
            # Execute trade
            success, trade_result = self.oanda.execute_trade(
                instrument=epic,
                units=units,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            if success:
                # Add additional trade info
                trade_result.update({
                    "risk_percent": risk_percent,
                    "risk_reward": trade.get("risk_reward"),
                    "pattern": trade.get("pattern"),
                    "reasoning": trade.get("reasoning"),
                    "stop_management": trade.get("stop_management", []),
                    "stop_loss": stop_loss
                })
                
                # Log the trade
                self.log_trade(trade_result)
                trades_executed += 1
            else:
                logger.warning(f"Failed to execute trade: {trade_result.get('reason', 'Unknown error')}")
                print(f"\n{Fore.RED}⚠️  TRADE EXECUTION FAILED{Style.RESET_ALL}\n  {epic} {direction} @ {entry_price}\n  Reason: {trade_result.get('reason', 'Unknown error')}\n")
        
        # Process position actions
        for action in execution_result.get("position_actions", []):
            action_type = action.get("action_type")
            epic = action.get("epic")
            
            if action_type == "CLOSE":
                # Find position
                deal_id = action.get("dealId")
                position = next((p for p in positions if p.get("dealId") == deal_id), None)
                
                if position:
                    success, close_result = self.oanda.close_position(epic)
                    if success:
                        # Log the close
                        close_result.update({
                            "reason": action.get("reason"),
                            "deal_id": deal_id
                        })
                        self.log_trade(close_result)
                    else:
                        logger.warning(f"Failed to close position: {close_result.get('reason', 'Unknown error')}")
            
            elif action_type == "UPDATE_STOP":
                # Update stop loss
                new_level = action.get("new_level")
                deal_id = action.get("dealId")
                position = next((p for p in positions if p.get("dealId") == deal_id), None)
                
                if position:
                    self.update_stop_loss(position, new_level)
        
        return trades_executed > 0
    
    def manage_positions(self):
        """Manage existing positions (trail stops, move to breakeven)"""
        positions = self.get_positions()
        
        if not positions:
            return
            
        # Get current prices
        current_prices = {}
        for pair in self.config["pairs"]:
            price = self.oanda.get_price(pair)
            if price:
                current_prices[pair] = price
        
        for position in positions:
            epic = position.get("epic")
            direction = position.get("direction")
            entry_level = float(position.get("level", 0))
            
            # Skip if we don't have price data
            if epic not in current_prices:
                continue
                
            current_price = current_prices[epic]
            current_bid = float(current_price.get("bid", 0))
            current_ask = float(current_price.get("offer", current_price.get("ask", 0)))
            
            # Get current price based on direction
            price_now = current_bid if direction == "SELL" else current_ask
            
            # Check for breakeven opportunity
            pip_size = 0.01 if "JPY" in epic else 0.0001
            breakeven_pips = self.config["breakeven_move_pips"].get(epic, 15)
            
            # Calculate price movement
            if direction == "BUY":
                pips_moved = (price_now - entry_level) / pip_size
            else: # SELL
                pips_moved = (entry_level - price_now) / pip_size
                
            # If moved enough pips, move stop to breakeven if it's below entry
            if pips_moved >= breakeven_pips:
                current_stop = float(position.get("stop_level", 0))
                
                # Check if stop needs moving to breakeven
                if (direction == "BUY" and current_stop < entry_level) or \
                   (direction == "SELL" and current_stop > entry_level):
                    
                    # Move stop to breakeven
                    self.update_stop_loss(position, entry_level)
    
    def display_status(self, account_info, positions, market_data):
        """Display current system status in the terminal"""
        print("\n" + "="*80)
        print(f"{Fore.CYAN}FOREX TRADING SYSTEM STATUS{Style.RESET_ALL}".center(80))
        print("="*80)
        
        # Account summary
        balance = float(account_info.get("balance", 0))
        currency = account_info.get("currency", "USD")
        
        print(f"\n{Fore.YELLOW}Account Summary:{Style.RESET_ALL}")
        print(f"  Balance: {balance:.2f} {currency}")
        print(f"  Daily P&L: {self.daily_pnl['total_pnl_pct']:.2f}% ({self.daily_pnl['realized_pnl']:.2f} realized, {self.daily_pnl['unrealized_pnl']:.2f} unrealized)")
        
        # Performance
        win_count = self.trading_memory["performance"]["win_count"]
        loss_count = self.trading_memory["performance"]["loss_count"]
        total_trades = win_count + loss_count
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
        
        print(f"\n{Fore.YELLOW}Performance:{Style.RESET_ALL}")
        print(f"  Win Rate: {win_rate:.1f}% ({win_count}/{total_trades} trades)")
        print(f"  Largest Win: {self.trading_memory['performance']['largest_win_pct']:.2f}%")
        print(f"  Largest Loss: {self.trading_memory['performance']['largest_loss_pct']:.2f}%")
        
        # Open positions
        print(f"\n{Fore.YELLOW}Open Positions ({len(positions)}):{Style.RESET_ALL}")
        if positions:
            position_data = []
            for p in positions:
                position_data.append([
                    p.get("epic"),
                    p.get("direction"),
                    p.get("level"),
                    p.get("profit"),
                    p.get("stop_level")
                ])
            print(tabulate(position_data, headers=["Pair", "Direction", "Entry", "P&L", "Stop Loss"]))
        else:
            print("  No open positions")
        
        # Market regimes
        print(f"\n{Fore.YELLOW}Market Regimes:{Style.RESET_ALL}")
        for pair, regime in self.market_regime.items():
            color = Fore.GREEN if "uptrend" in regime else Fore.RED if "downtrend" in regime else Fore.YELLOW
            print(f"  {pair}: {color}{regime}{Style.RESET_ALL}")
        
        # Current prices
        print(f"\n{Fore.YELLOW}Current Prices:{Style.RESET_ALL}")
        for pair in self.config["pairs"]:
            if pair in market_data and "current" in market_data[pair]:
                current = market_data[pair]["current"]
                bid = current.get("bid", 0)
                ask = current.get("offer", current.get("ask", 0))
                print(f"  {pair}: {bid}/{ask}")
        
        # Budget status
        remaining = self.session_usage["budget"] - self.session_usage["spent"]
        used_pct = (self.session_usage["spent"] / self.session_usage["budget"]) * 100
        print(f"\n{Fore.YELLOW}LLM Budget:{Style.RESET_ALL}")
        print(f"  ${remaining:.2f} remaining ({used_pct:.1f}% used)")
        
        # Recent recommendations for human
        if self.trading_memory.get("human_recommendations"):
            print(f"\n{Fore.GREEN}Recent Recommendations for You:{Style.RESET_ALL}")
            for rec in self.trading_memory["human_recommendations"][-3:]:
                print(f"  • {rec}")
                
        print("\n" + "="*80 + "\n")
    
    def run_trading_cycle(self):
        """Run a complete trading cycle"""
        logger.info("Starting trading cycle")
        
        try:
            # Check if we have enough budget
            remaining_budget = self.session_usage["budget"] - self.session_usage["spent"]
            if remaining_budget < 1.0:
                logger.warning(f"Insufficient budget remaining (${remaining_budget:.2f}). Skipping cycle.")
                return False
            
            # Collect market data
            market_data = self.get_market_data()
            
            # Get account and position data
            account_info = self.get_account_info()
            positions = self.get_positions()
            
            # Update daily P&L tracking
            self.update_daily_pnl(account_info, positions)
            
            # Get recent trades
            recent_trades = self.get_recent_trades()
            
            # 1. Run Analysis Agent
            analysis_result = self.analysis_agent.run(
                market_data=market_data,
                account_data=account_info,
                positions=positions,
                recent_trades=recent_trades,
                config=self.config,
                trading_memory=self.trading_memory,
                market_regime=self.market_regime
            )
            
            # Store analysis feedback
            if analysis_result and "self_improvement" in analysis_result:
                self.trading_memory["feedback"]["analysis"] = analysis_result["self_improvement"]
                
            # 2. Run Execution Agent if analysis produced results
            execution_result = None
            if analysis_result and "analysis_results" in analysis_result:
                analysis_results = analysis_result.get("analysis_results", [])
                
                # Only proceed if we have valid analysis results
                if analysis_results:
                    execution_result = self.execution_agent.run(
                        analysis_results=analysis_results,
                        market_data=market_data,
                        account_data=account_info,
                        positions=positions,
                        recent_trades=recent_trades,
                        config=self.config,
                        trading_memory=self.trading_memory,
                        market_regime=self.market_regime
                    )
                    
                    # Store execution feedback
                    if execution_result and "self_improvement" in execution_result:
                        self.trading_memory["feedback"]["execution"] = execution_result["self_improvement"]
                    
                    # 3. Execute trades
                    self.execute_trades(execution_result)
            
            # 4. Manage existing positions
            self.manage_positions()
            
            # 5. Run Review Agent
            if analysis_result or execution_result:
                review_result = self.review_agent.run(
                    analysis_result=analysis_result,
                    execution_result=execution_result,
                    market_data=market_data,
                    account_data=account_info,
                    positions=positions,
                    recent_trades=recent_trades,
                    config=self.config,
                    trading_memory=self.trading_memory,
                    market_regime=self.market_regime
                )
                
                # Store review feedback and recommendations
                if review_result:
                    if "feedback" in review_result:
                        self.trading_memory["feedback"]["review"] = review_result["feedback"]
                    
                    if "human_recommendations" in review_result:
                        self.trading_memory["human_recommendations"] = (
                            self.trading_memory.get("human_recommendations", []) + 
                            review_result["human_recommendations"]
                        )[-10:]  # Keep only the last 10 recommendations
                    
                    if "learning_insights" in review_result:
                        for key, value in review_result["learning_insights"].items():
                            if key in self.trading_memory["learning"]:
                                self.trading_memory["learning"][key].update(value)
            
            # Update last cycle timestamp
            self.trading_memory["last_cycle"] = datetime.now(timezone.utc).isoformat()
            self.save_memory()
            
            # Display system status
            self.display_status(account_info, positions, market_data)
            
            # Log budget status
            remaining = self.session_usage["budget"] - self.session_usage["spent"]
            used_pct = (self.session_usage["spent"] / self.session_usage["budget"]) * 100
            logger.info(f"Budget status: ${remaining:.2f} remaining ({used_pct:.1f}% used)")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}", exc_info=True)
            return False
    
    def run(self):
        """Main trading loop"""
        logger.info("Starting Dual-Pair LLM Forex Trading System")
        print("\n" + "="*80)
        print(f"{Fore.CYAN}STARTING DUAL-PAIR LLM FOREX TRADING SYSTEM{Style.RESET_ALL}".center(80))
        print("="*80)
        
        print(f"\n{Fore.YELLOW}Trading Pairs:{Style.RESET_ALL} {', '.join(self.config['pairs'])}")
        print(f"{Fore.YELLOW}Primary:{Style.RESET_ALL} {self.config['primary_pair']}, {Fore.YELLOW}Secondary:{Style.RESET_ALL} {self.config['secondary_pair']}")
        print(f"{Fore.YELLOW}Budget:{Style.RESET_ALL} ${self.config['daily_budget']:.2f}")
        
        # Initial account info
        account = self.get_account_info()
        initial_balance = float(account.get("balance", 0))
        currency = account.get("currency", "USD")
        print(f"{Fore.YELLOW}Account Balance:{Style.RESET_ALL} {initial_balance:.2f} {currency}")
        
        # Set initial daily P&L tracking
        self.daily_pnl["start_balance"] = initial_balance
        self.daily_pnl["current_balance"] = initial_balance
        
        # Trading loop
        while True:
            try:
                # Run a trading cycle
                self.run_trading_cycle()
                
                # Sleep between cycles
                cycle_minutes = self.config["cycle_minutes"]
                print(f"\n{Fore.BLUE}⏱️  Cycle complete. Sleeping for {cycle_minutes} minutes.{Style.RESET_ALL}")
                time.sleep(cycle_minutes * 60)
                
            except KeyboardInterrupt:
                logger.info("Trading system stopped by user.")
                print(f"\n{Fore.RED}TRADING SYSTEM STOPPED BY USER{Style.RESET_ALL}")
                break
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                print(f"\n{Fore.RED}ERROR IN MAIN LOOP: {e}{Style.RESET_ALL}")
                # Sleep shorter time on error
                time.sleep(60)

def main():
    # Check if OANDA credentials are set
    if not os.getenv("OANDA_API_TOKEN") or not os.getenv("OANDA_ACCOUNT_ID"):
        print(f"{Fore.RED}ERROR: OANDA API credentials not set. Please check your .env file.{Style.RESET_ALL}")
        print("Required variables: OANDA_API_TOKEN, OANDA_ACCOUNT_ID")
        return
        
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print(f"{Fore.RED}ERROR: OpenAI API key not set. Please check your .env file.{Style.RESET_ALL}")
        print("Required variable: OPENAI_API_KEY")
        return
        
    # Start trading system
    system = TradingSystem()
    system.run()

if __name__ == "__main__":
    main()