# Save as: src/caspar/tools/accounts.py

"""
Account Information Tool

Retrieves customer account information.
In production, this would connect to your CRM or user database.
"""

from datetime import datetime, timedelta
from pydantic import BaseModel
import random

from caspar.config import get_logger

logger = get_logger(__name__)


class CustomerAccount(BaseModel):
    """Customer account information."""
    
    customer_id: str
    email: str
    name: str
    phone: str | None = None
    member_since: str
    loyalty_tier: str  # bronze, silver, gold, platinum
    loyalty_points: int
    total_orders: int
    total_spent: float
    default_shipping_address: dict | None = None
    payment_methods_on_file: int
    email_verified: bool = True
    two_factor_enabled: bool = False


class AccountTool:
    """
    Tool for retrieving customer account information.
    
    In production, this would query your CRM or database.
    """
    
    def __init__(self):
        # Mock customer database
        self._mock_accounts = self._generate_mock_accounts()
    
    def _generate_mock_accounts(self) -> dict[str, CustomerAccount]:
        """Generate mock customer data."""
        
        tiers = ["bronze", "silver", "gold", "platinum"]
        
        accounts = {}
        
        mock_customers = [
            ("CUST-1000", "john.doe@email.com", "John Doe"),
            ("CUST-1001", "jane.smith@email.com", "Jane Smith"),
            ("CUST-1002", "bob.wilson@email.com", "Bob Wilson"),
            ("CUST-1003", "alice.jones@email.com", "Alice Jones"),
            ("CUST-1004", "charlie.brown@email.com", "Charlie Brown"),
        ]
        
        for i, (cust_id, email, name) in enumerate(mock_customers):
            # Vary the data for each customer
            tier_index = min(i, len(tiers) - 1)
            orders = (i + 1) * 5
            spent = orders * random.uniform(100, 500)
            
            accounts[cust_id] = CustomerAccount(
                customer_id=cust_id,
                email=email,
                name=name,
                phone=f"+1-555-{1000 + i:04d}" if i % 2 == 0 else None,
                member_since=(datetime.now() - timedelta(days=365 * (i + 1))).strftime("%Y-%m-%d"),
                loyalty_tier=tiers[tier_index],
                loyalty_points=int(spent * 10),
                total_orders=orders,
                total_spent=round(spent, 2),
                default_shipping_address={
                    "street": f"{100 + i} Main Street",
                    "city": "Anytown",
                    "state": "CA",
                    "zip": f"9{1000 + i}",
                    "country": "USA"
                } if i % 2 == 0 else None,
                payment_methods_on_file=min(i + 1, 3),
                email_verified=True,
                two_factor_enabled=i > 2,
            )
        
        return accounts
    
    def get_account(self, customer_id: str) -> CustomerAccount | None:
        """
        Retrieve account information by customer ID.
        
        Args:
            customer_id: The customer's ID
            
        Returns:
            CustomerAccount if found, None otherwise
        """
        logger.info("account_lookup", customer_id=customer_id)
        
        account = self._mock_accounts.get(customer_id)
        
        if account is None:
            logger.warning("account_not_found", customer_id=customer_id)
            return None
        
        logger.info("account_found", customer_id=customer_id, tier=account.loyalty_tier)
        return account
    
    def get_account_by_email(self, email: str) -> CustomerAccount | None:
        """Look up account by email address."""
        email_lower = email.lower()
        for account in self._mock_accounts.values():
            if account.email.lower() == email_lower:
                return account
        return None
    
    def format_account_summary(self, account: CustomerAccount) -> str:
        """Format account info for display to customer."""
        
        tier_emoji = {
            "bronze": "🥉",
            "silver": "🥈",
            "gold": "🥇",
            "platinum": "💎"
        }
        
        lines = [
            f"**Account Summary for {account.name}**",
            "",
            f"Member Since: {account.member_since}",
            f"Loyalty Status: {tier_emoji.get(account.loyalty_tier, '')} {account.loyalty_tier.title()}",
            f"Loyalty Points: {account.loyalty_points:,}",
            "",
            f"Total Orders: {account.total_orders}",
            f"Total Spent: ${account.total_spent:,.2f}",
            "",
            f"Email: {account.email} {'✓ Verified' if account.email_verified else '⚠ Not verified'}",
        ]
        
        if account.phone:
            lines.append(f"Phone: {account.phone}")
        
        if account.default_shipping_address:
            addr = account.default_shipping_address
            lines.append(f"\nDefault Shipping Address:")
            lines.append(f"  {addr['street']}")
            lines.append(f"  {addr['city']}, {addr['state']} {addr['zip']}")
        
        lines.append(f"\nPayment Methods: {account.payment_methods_on_file} on file")
        lines.append(f"Two-Factor Auth: {'✓ Enabled' if account.two_factor_enabled else 'Not enabled'}")
        
        return "\n".join(lines)


# Singleton instance
_account_tool: AccountTool | None = None


def get_account_tool() -> AccountTool:
    """Get or create the account tool instance."""
    global _account_tool
    if _account_tool is None:
        _account_tool = AccountTool()
    return _account_tool


def get_account_info(customer_id: str) -> dict:
    """
    Convenience function to get account information.
    
    Returns a dict with account info or error message.
    """
    tool = get_account_tool()
    account = tool.get_account(customer_id)
    
    if account is None:
        return {
            "found": False,
            "error": f"Account {customer_id} not found."
        }
    
    return {
        "found": True,
        "account": account.model_dump(),
        "summary": tool.format_account_summary(account)
    }