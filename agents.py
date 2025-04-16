"""
Enhanced LLM Agents for Forex Trading
Performs market analysis, trade execution, and review with improved reasoning
"""

import json
import logging
import os
from datetime import datetime, timezone
import openai

logger = logging.getLogger("ForexTrader")

class BaseAgent:
    """Base class for LLM agents"""
    
    def __init__(self, model, provider="OpenAI", budget_manager=None):
        self.model = model
        self.provider = provider
        self.budget_manager = budget_manager
        
        # Set up API key based on provider
        if provider == "OpenAI":
            openai.api_key = os.getenv("OPENAI_API_KEY")
        elif provider == "Claude":
            # Configure for Anthropic API
            self.api_key = os.getenv("CLAUDE_API_KEY")
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    def _call_llm(self, system_message, user_message, temperature=0.3):
        """Call LLM API with appropriate provider"""
        start_time = datetime.now(timezone.utc)
        
        try:
            # Build parameters for API call
            if self.provider == "OpenAI":
                response = openai.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=temperature,
                    response_format={"type": "json_object"},
                    timeout=60
                )
                
                # Calculate cost (simplified approximation)
                tokens_in = response.usage.prompt_tokens
                tokens_out = response.usage.completion_tokens
                
                if "gpt-4" in self.model:
                    cost = (tokens_in * 0.03 + tokens_out * 0.06) / 1000
                else:  # GPT-3.5
                    cost = (tokens_in * 0.0015 + tokens_out * 0.002) / 1000
                
                # Track cost if budget manager provided
                if self.budget_manager:
                    self.budget_manager(cost, self.__class__.__name__)
                
                # Parse response
                content = response.choices[0].message.content
                try:
                    result = json.loads(content)
                    return result
                except json.JSONDecodeError:
                    # Try to extract JSON if wrapped in code blocks
                    import re
                    json_match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
                    if json_match:
                        try:
                            return json.loads(json_match.group(1))
                        except:
                            pass
                    
                    # If still can't parse, return error
                    logger.error(f"Failed to parse JSON from response: {content[:100]}...")
                    return {"error": "Failed to parse response as JSON"}
                    
            elif self.provider == "Claude":
                # Implementation for Anthropic's Claude API would go here
                pass
                
        except Exception as e:
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            logger.error(f"LLM API error after {duration:.1f}s: {e}")
            return {"error": str(e)}
    
    def run(self, **kwargs):
        """Run the agent (must be implemented by subclasses)"""
        raise NotImplementedError("Subclasses must implement run method")


class AnalysisAgent(BaseAgent):
    """Agent that analyzes market opportunities with enhanced technical analysis"""
    
    def run(self, market_data, account_data, positions, recent_trades, config, trading_memory, market_regime):
        """Run the analysis agent to find trading opportunities"""
        logger.info("Running Analysis Agent")
        
        try:
            # Create market summary for prompt
            market_summary = self._format_market_data(market_data, config["pairs"])
            
            # Create position summary
            position_summary = self._format_positions(positions)
            
            # Create trade history summary
            trade_history = self._format_trade_history(recent_trades)
            
            # Get pattern effectiveness from memory
            pattern_effectiveness = {
                "successful": trading_memory.get("learning", {}).get("successful_patterns", {}),
                "failed": trading_memory.get("learning", {}).get("failed_patterns", {})
            }
            
            # Create prompt
            system_message = self._get_system_message()
            user_message = self._get_user_message(
                market_summary=market_summary,
                position_summary=position_summary,
                trade_history=trade_history,
                account_data=account_data,
                config=config,
                trading_memory=trading_memory,
                market_regime=market_regime,
                pattern_effectiveness=pattern_effectiveness
            )
            
            # Call LLM API
            result = self._call_llm(system_message, user_message)
            
            if result and "analysis_results" in result:
                logger.info(f"Analysis found {len(result['analysis_results'])} trading opportunities")
                return result
            else:
                if "error" in result:
                    logger.error(f"Analysis agent error: {result['error']}")
                else:
                    logger.warning("Analysis agent produced no results")
                return None
                
        except Exception as e:
            logger.error(f"Error in analysis agent: {e}", exc_info=True)
            return None
    
    def _format_market_data(self, market_data, pairs):
        """Format market data for the prompt"""
        formatted = {}
        
        for pair in pairs:
            if pair not in market_data:
                continue
                
            pair_data = market_data[pair]
            
            # Current price
            current = pair_data.get("current", {})
            formatted[pair] = {
                "current": {
                    "bid": current.get("bid"),
                    "ask": current.get("offer", current.get("ask")),
                    "timestamp": current.get("timestamp")
                }
            }
            
            # Recent candles for each timeframe
            for tf, candles in pair_data.items():
                if tf == "current" or not candles:
                    continue
                    
                # Add the most recent candles
                formatted[pair][tf] = []
                for candle in candles[-7:]:  # Last 7 candles
                    if isinstance(candle, dict) and "mid" in candle:
                        # OANDA format
                        mid = candle.get("mid", {})
                        formatted[pair][tf].append({
                            "timestamp": candle.get("time"),
                            "open": float(mid.get("o", 0)),
                            "high": float(mid.get("h", 0)),
                            "low": float(mid.get("l", 0)),
                            "close": float(mid.get("c", 0)),
                            "volume": int(candle.get("volume", 0))
                        })
                    elif isinstance(candle, dict):
                        # Already in our format
                        formatted[pair][tf].append(candle)
        
        return formatted
    
    def _format_positions(self, positions):
        """Format positions for the prompt"""
        if not positions:
            return []
            
        formatted = []
        for position in positions:
            formatted.append({
                "epic": position.get("epic"),
                "direction": position.get("direction"),
                "size": position.get("size"),
                "entry_price": position.get("level"),
                "current_profit": position.get("profit"),
                "stop_level": position.get("stop_level", 0)
            })
            
        return formatted
    
    def _format_trade_history(self, trades):
        """Format trade history for the prompt"""
        if not trades:
            return []
            
        formatted = []
        for trade in trades:
            # Only include relevant info
            formatted.append({
                "timestamp": trade.get("timestamp"),
                "epic": trade.get("epic"),
                "direction": trade.get("direction"),
                "action_type": trade.get("action_type", "OPEN"),
                "entry_price": trade.get("entry_price"),
                "outcome": trade.get("outcome", ""),
                "pattern": trade.get("pattern", ""),
                "risk_reward": trade.get("risk_reward"),
                "reasoning": trade.get("reasoning", "")
            })
            
        return formatted
    
    def _get_system_message(self):
        """Get system message for the analysis agent"""
        return """You are an expert forex trading analyst specializing in technical analysis for EUR/USD and EUR/GBP pairs. Your expertise involves multi-timeframe analysis, identifying high-probability trade setups, and providing detailed trade plans with precise entry, exit, and risk management parameters.

Key responsibilities:
1. Analyze all available timeframes (M5, M15, H1, H4, D1, W1) to identify the strongest trading opportunities
2. Apply multiple technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
3. Identify key support/resistance levels and chart patterns
4. Determine market regime (trending, ranging, volatile) and adjust strategy accordingly
5. Calculate precise entry zones, stop losses, and multiple take profit targets
6. Provide in-depth reasoning and analysis for each trade recommendation
7. Consider previous trade performance and pattern effectiveness in your analysis
8. Adapt strategy based on prevailing market conditions for each currency pair

You provide detailed explanations of your thought process, ensuring your recommendations are clear, justified by technical evidence, and include thorough risk management parameters.

Respond with JSON containing your complete market analysis and trading recommendations."""
    
    def _get_user_message(self, market_summary, position_summary, trade_history, account_data, config, trading_memory, market_regime, pattern_effectiveness):
        """Get user message for the analysis agent"""
        # Extract recent performance metrics
        performance = trading_memory.get("performance", {})
        win_count = performance.get("win_count", 0)
        loss_count = performance.get("loss_count", 0)
        total_trades = win_count + loss_count
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
        
        # Format prior feedback if available
        feedback = trading_memory.get("feedback", {}).get("analysis", "")
        feedback_section = f"\n## Previous Feedback\n{feedback}" if feedback else ""
        
        # Format pattern effectiveness
        pattern_section = "\n## Pattern Effectiveness\n"
        
        if pattern_effectiveness["successful"]:
            pattern_section += "### Successful Patterns\n"
            for pattern, count in sorted(pattern_effectiveness["successful"].items(), key=lambda x: x[1], reverse=True):
                pattern_section += f"- {pattern}: {count} successful trades\n"
                
        if pattern_effectiveness["failed"]:
            pattern_section += "\n### Failed Patterns\n"
            for pattern, count in sorted(pattern_effectiveness["failed"].items(), key=lambda x: x[1], reverse=True):
                pattern_section += f"- {pattern}: {count} failed trades\n"
        
        # Format market regime
        regime_section = "\n## Market Regimes\n"
        for pair, regime in market_regime.items():
            regime_section += f"- {pair}: {regime}\n"
        
        return f"""
# Enhanced Market Analysis Task

## Account Status
- Balance: {account_data.get('balance')} {account_data.get('currency')}
- Win Rate: {win_rate:.1f}% ({win_count}/{total_trades} trades)
- Open Positions: {len(position_summary)}

## Trading Configuration
- Primary Pair: {config['primary_pair']}
- Secondary Pair: {config['secondary_pair']}
- Min Quality Score: {config['min_quality_score']}/10
- Min Risk-Reward: {config['min_risk_reward']}
- Base Risk: {config['base_risk_percent']}%
{regime_section}

## Current Positions
{json.dumps(position_summary, indent=2)}

## Recent Trades
{json.dumps(trade_history[:5], indent=2)}
{pattern_section}
{feedback_section}

## Current Market Data
{json.dumps(market_summary, indent=2)}

## Analysis Instructions
1. Perform multi-timeframe analysis on EUR/USD and EUR/GBP using all available timeframes
2. Apply technical indicators including:
   - Moving Averages (SMA and EMA)
   - RSI (including divergence analysis)
   - MACD (signal line crossovers and histogram analysis)
   - Bollinger Bands (including squeeze patterns)
   - Support/Resistance levels across multiple timeframes

3. For each currency pair, determine:
   - Overall trend direction on each timeframe
   - Key support and resistance levels
   - Notable chart patterns forming
   - Potential entry zones with confluence from multiple indicators
   - Logical stop loss levels based on market structure
   - Multiple take profit targets with probability estimates

4. Consider market regime for each pair and adapt your analysis accordingly:
   - In trending markets: focus on pullbacks to moving averages
   - In ranging markets: focus on support/resistance bounces
   - In volatile markets: widen stops and prioritize higher probability setups

5. For each trading opportunity, provide:
   - Precise entry zone with ideal price and acceptable range
   - Stop loss with clear reasoning based on market structure
   - Multiple take profit targets with probabilities
   - Quality score (1-10) based on pattern clarity and confluence
   - Risk-reward calculation
   - Step-by-step reasoning explaining the opportunity

6. Make sure each trade has a specific named pattern for tracking
7. Focus on opportunities with quality scores of {config['min_quality_score']}+ and risk-reward of at least {config['min_risk_reward']}
8. If no high-quality setups exist, clearly state that no trades meet the criteria

## Response Format
Respond with a JSON object containing:
1. "market_assessment" object with detailed market analysis for each pair
2. "analysis_results" array with detailed analysis for each trading opportunity
3. "self_improvement" object with reflections on your analysis process
4. "reasoning_process" object explaining your analytical approach in depth
"""


class ExecutionAgent(BaseAgent):
    """Agent that makes final trading decisions with enhanced reasoning"""
    
    def run(self, analysis_results, market_data, account_data, positions, recent_trades, config, trading_memory, market_regime):
        """Run the execution agent to make trading decisions"""
        logger.info("Running Execution Agent")
        
        try:
            # Create market summary
            market_summary = {}
            for pair in config["pairs"]:
                if pair in market_data and "current" in market_data[pair]:
                    current = market_data[pair]["current"]
                    market_summary[pair] = {
                        "bid": current.get("bid"),
                        "ask": current.get("offer", current.get("ask"))
                    }
            
            # Create position summary
            position_summary = []
            for position in positions:
                position_summary.append({
                    "epic": position.get("epic"),
                    "direction": position.get("direction"),
                    "size": position.get("size"),
                    "entry_price": position.get("level"),
                    "current_profit": position.get("profit"),
                    "dealId": position.get("dealId"),
                    "stop_level": position.get("stop_level", 0)
                })
            
            # Format market regime
            regime_data = {pair: regime for pair, regime in market_regime.items()}
            
            # Create prompt
            system_message = self._get_system_message()
            user_message = self._get_user_message(
                analysis_results=analysis_results,
                market_summary=market_summary,
                position_summary=position_summary,
                recent_trades=recent_trades[:5],
                account_data=account_data,
                config=config,
                trading_memory=trading_memory,
                market_regime=regime_data
            )
            
            # Call LLM API
            result = self._call_llm(system_message, user_message)
            
            if result and "trade_actions" in result:
                logger.info(f"Execution agent decided on {len(result['trade_actions'])} trade actions and {len(result.get('position_actions', []))} position updates")
                return result
            else:
                if "error" in result:
                    logger.error(f"Execution agent error: {result['error']}")
                else:
                    logger.warning("Execution agent produced no results")
                return None
                
        except Exception as e:
            logger.error(f"Error in execution agent: {e}", exc_info=True)
            return None
    
    def _get_system_message(self):
        """Get system message for the execution agent"""
        return """You are an expert forex trading decision maker specializing in EUR/USD and EUR/GBP pairs. Your role is to review detailed market analysis and make final trading decisions with precise execution parameters. You excel at portfolio management, risk allocation, and ensuring each trade meets strict quality criteria.

Key responsibilities:
1. Evaluate analysis quality and determine which trade opportunities to execute
2. Assign appropriate risk percentages based on setup quality and conviction
3. Determine exact entry prices, stop losses, and take profit levels
4. Manage overall portfolio exposure and risk distribution
5. Monitor existing positions and recommend updates (close or trail stops)
6. Adapt decision-making to current market regimes
7. Maintain careful position sizing and correlation management
8. Provide detailed reasoning for all execution decisions

You are decisive and precise, providing exact numerical values for all trade parameters. You maintain strict risk management rules and ensure the trading system operates within defined parameters.

Respond with JSON containing your complete trading recommendations and detailed reasoning."""
    
    def _get_user_message(self, analysis_results, market_summary, position_summary, recent_trades, account_data, config, trading_memory, market_regime):
        """Get user message for the execution agent"""
        # Extract performance metrics
        performance = trading_memory.get("performance", {})
        win_count = performance.get("win_count", 0)
        loss_count = performance.get("loss_count", 0)
        total_trades = win_count + loss_count
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
        
        # Format prior feedback if available
        feedback = trading_memory.get("feedback", {}).get("execution", "")
        feedback_section = f"\n## Previous Feedback\n{feedback}" if feedback else ""
        
        # Format market regime
        regime_section = "\n## Market Regimes\n"
        for pair, regime in market_regime.items():
            regime_section += f"- {pair}: {regime}\n"
        
        return f"""
# Trading Execution Task

## MANDATORY TRADE PARAMETERS
CRITICAL: Each trade MUST include the following numerical values:
- Exact entry price (number, not null)
- Exact stop loss level (number, not null)
- At least one take profit level (number)
- Risk percentage between 1-2%

Trades missing ANY of these parameters will be rejected.

## Account Status
- Balance: {account_data.get('balance')} {account_data.get('currency')}
- Win Rate: {win_rate:.1f}% ({win_count}/{total_trades} trades)
- Open Positions: {len(position_summary)}
{regime_section}

## Trading Configuration
- Primary Pair: {config['primary_pair']}
- Secondary Pair: {config['secondary_pair']}
- Min Quality Score: {config['min_quality_score']}/10
- Min Risk-Reward: {config['min_risk_reward']}
- Base Risk: {config['base_risk_percent']}%
- Max Risk Per Pair: {config['max_risk_per_pair']}%
- Max Total Risk: {config['max_total_risk']}%
- Max Positions Per Pair: {config['max_positions_per_pair']}
- Max Total Positions: {config['max_total_positions']}
- Min Total Positions: {config['min_total_positions']}

## Current Market Prices
{json.dumps(market_summary, indent=2)}

## Current Positions
{json.dumps(position_summary, indent=2)}

## Recent Trades
{json.dumps(recent_trades, indent=2)}
{feedback_section}

## Analysis Results
{json.dumps(analysis_results, indent=2)}

## Execution Instructions
1. Review all analysis results and decide which trades to execute based on:
   - Analysis quality (minimum score {config['min_quality_score']})
   - Risk-reward ratio (minimum {config['min_risk_reward']})
   - Market regime compatibility (is the setup appropriate for current conditions?)
   - Portfolio diversification (avoid too much exposure to one currency)

2. For each trade:
   - Assign risk percentage (1-2%) based on setup quality and conviction
   - Specify exact entry price, stop loss, and take profit levels
   - Define stop management strategy
   - Provide detailed reasoning for execution decision

3. Evaluate existing positions:
   - Recommend closing any positions where the setup is no longer valid
   - Suggest updates to stop losses for positions in profit
   - Provide reasoning for each position action

4. Maintain proper risk distribution:
   - No more than {config['max_positions_per_pair']} positions per pair
   - No more than {config['max_total_positions']} total positions
   - No more than {config['max_risk_per_pair']}% risk on a single pair
   - No more than {config['max_total_risk']}% total account risk
   - Ensure at least {config['min_total_positions']} position when possible

5. Adapt execution to market regime:
   - In trending markets: prioritize trend-following setups
   - In ranging markets: prioritize reversal setups at range boundaries
   - In volatile markets: reduce position sizes and widen stops

## Response Format
Respond with a JSON object containing:
1. "trade_actions" array with new trades to execute:
   - "action_type": "OPEN" (required)
   - "epic": Currency pair code (required)
   - "direction": "BUY" or "SELL" (required)
   - "entry_price": Exact numerical price for entry (required)
   - "initial_stop_loss": Exact numerical price for stop loss (required)
   - "take_profit_levels": Array of numerical prices [level1, level2, ...] (required)
   - "risk_percent": Percentage of account to risk (required, 1-2%)
   - "risk_reward": Expected R:R ratio (required)
   - "pattern": Pattern being traded
   - "reasoning": Detailed reasoning for this execution decision

2. "position_actions" array with actions for existing positions:
   - "action_type": "CLOSE" or "UPDATE_STOP"
   - "epic": Currency pair code
   - "dealId": Deal identifier
   - "new_level": New level for stop loss (if UPDATE_STOP)
   - "reason": Detailed reasoning for the action

3. "risk_assessment" object with your portfolio risk evaluation:
   - "current_exposure": Overall market exposure assessment
   - "pair_allocations": Risk allocation per currency pair
   - "correlation_management": How you're managing correlations
   - "strategy_adaptation": How you've adapted to market regimes

4. "execution_reasoning" object with detailed explanation of your decision process:
   - "trade_selection_process": How you chose which trades to execute
   - "risk_allocation_process": How you determined risk percentages
   - "position_management_process": How you decided on position actions

5. "self_improvement" object with your reflections and learnings:
   - "execution_effectiveness": Assessment of your decision making
   - "improvement_areas": Specific areas you could improve
   - "strategy_adjustments": Suggested adjustments to trading strategy
"""


class ReviewAgent(BaseAgent):
    """Agent that reviews trading decisions and provides feedback and recommendations"""
    
    def run(self, analysis_result, execution_result, market_data, account_data, positions, recent_trades, config, trading_memory, market_regime):
        """Run the review agent to evaluate decisions and provide feedback"""
        logger.info("Running Review Agent")
        
        try:
            # Create market summary
            market_summary = {}
            for pair in config["pairs"]:
                if pair in market_data and "current" in market_data[pair]:
                    current = market_data[pair]["current"]
                    market_summary[pair] = {
                        "bid": current.get("bid"),
                        "ask": current.get("offer", current.get("ask"))
                    }
            
            # Create position summary
            position_summary = []
            for position in positions:
                position_summary.append({
                    "epic": position.get("epic"),
                    "direction": position.get("direction"),
                    "size": position.get("size"),
                    "entry_price": position.get("level"),
                    "current_profit": position.get("profit"),
                    "dealId": position.get("dealId"),
                    "stop_level": position.get("stop_level", 0)
                })
            
            # Format market regime
            regime_data = {pair: regime for pair, regime in market_regime.items()}
            
            # Get system performance data
            performance = trading_memory.get("performance", {})
            
            # Get pattern effectiveness
            pattern_effectiveness = {
                "successful": trading_memory.get("learning", {}).get("successful_patterns", {}),
                "failed": trading_memory.get("learning", {}).get("failed_patterns", {})
            }
            
            # Create prompt
            system_message = self._get_system_message()
            user_message = self._get_user_message(
                analysis_result=analysis_result,
                execution_result=execution_result,
                market_summary=market_summary,
                position_summary=position_summary,
                recent_trades=recent_trades[:10],
                account_data=account_data,
                config=config,
                performance=performance,
                pattern_effectiveness=pattern_effectiveness,
                market_regime=regime_data
            )
            
            # Call LLM API
            result = self._call_llm(system_message, user_message, temperature=0.4)
            
            if "error" in result:
                logger.error(f"Review agent error: {result['error']}")
                return None
                
            logger.info("Review agent completed analysis")
            return result
                
        except Exception as e:
            logger.error(f"Error in review agent: {e}", exc_info=True)
            return None
    
    def _get_system_message(self):
        """Get system message for the review agent"""
        return """You are an expert forex trading reviewer and advisor who evaluates trading decisions and provides detailed feedback and recommendations. Your purpose is to analyze the effectiveness of trading strategies, identify learning opportunities, and suggest improvements to the trading system.

Key responsibilities:
1. Evaluate the quality of market analysis and trading decisions
2. Identify strengths and weaknesses in the trading approach
3. Recommend specific improvements to the trading strategy
4. Provide actionable insights for the human trader
5. Track pattern effectiveness and strategy performance
6. Identify emerging market conditions that require adaptation
7. Suggest adjustments to risk management parameters
8. Generate concise but insightful recommendations

You are thoughtful, balanced, and focused on continuous improvement. You provide candid feedback while maintaining a constructive approach. Your recommendations are specific, practical, and aimed at improving trading performance.

Respond with JSON containing your complete review, insights, and recommendations."""
    
    def _get_user_message(self, analysis_result, execution_result, market_summary, position_summary, recent_trades, account_data, config, performance, pattern_effectiveness, market_regime):
        """Get user message for the review agent"""
        # Calculate win rate
        win_count = performance.get("win_count", 0)
        loss_count = performance.get("loss_count", 0)
        total_trades = win_count + loss_count
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
        
        # Format pattern effectiveness
        pattern_section = "\n## Pattern Effectiveness\n"
        
        if pattern_effectiveness["successful"]:
            pattern_section += "### Successful Patterns\n"
            for pattern, count in sorted(pattern_effectiveness["successful"].items(), key=lambda x: x[1], reverse=True):
                pattern_section += f"- {pattern}: {count} successful trades\n"
                
        if pattern_effectiveness["failed"]:
            pattern_section += "\n### Failed Patterns\n"
            for pattern, count in sorted(pattern_effectiveness["failed"].items(), key=lambda x: x[1], reverse=True):
                pattern_section += f"- {pattern}: {count} failed trades\n"
        
        # Format market regime
        regime_section = "\n## Market Regimes\n"
        for pair, regime in market_regime.items():
            regime_section += f"- {pair}: {regime}\n"
        
        return f"""
# Trading Review and Feedback Task

## Account Status
- Balance: {account_data.get('balance')} {account_data.get('currency')}
- Win Rate: {win_rate:.1f}% ({win_count}/{total_trades} trades)
- Largest Win: {performance.get('largest_win_pct', 0):.2f}%
- Largest Loss: {performance.get('largest_loss_pct', 0):.2f}%
- Open Positions: {len(position_summary)}
{regime_section}

## Current Market Prices
{json.dumps(market_summary, indent=2)}

## Current Positions
{json.dumps(position_summary, indent=2)}

## Recent Trades
{json.dumps(recent_trades, indent=2)}
{pattern_section}

## Analysis Results
{json.dumps(analysis_result, indent=2) if analysis_result else "No analysis results"}

## Execution Results
{json.dumps(execution_result, indent=2) if execution_result else "No execution results"}

## Review Instructions
1. Evaluate the quality of the market analysis:
   - Are the identified patterns valid and well-supported?
   - Is the multi-timeframe analysis consistent and logical?
   - Are support/resistance levels accurately identified?
   - Is the risk-reward assessment realistic?

2. Assess the execution decisions:
   - Are the chosen trades appropriate given the analysis?
   - Is the risk allocation appropriate for each setup?
   - Are position management decisions sound?
   - Is there appropriate adaptation to market regimes?

3. Identify learning opportunities:
   - What patterns or setups are working best?
   - What timeframes are most reliable?
   - What improvements could be made to entry/exit criteria?
   - What adjustments to risk management might improve results?

4. Provide recommendations for the human trader:
   - What specific aspects of the system could be improved?
   - Are there market conditions to be watchful of?
   - What parameter adjustments might improve performance?
   - What additional data or analysis would help decision making?

## Response Format
Respond with a JSON object containing:
1. "analysis_review" object evaluating the analysis quality and accuracy
2. "execution_review" object evaluating the execution decisions
3. "strategy_assessment" object with overall trading strategy evaluation
4. "learning_insights" object with pattern and setup effectiveness
5. "human_recommendations" array with 3-5 specific recommendations for the human trader
6. "feedback" object with general improvement suggestions for the system
"""