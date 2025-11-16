"""EVE ESI API client utilities - simplified for demonstration"""

import httpx
from typing import List, Dict, Any, Optional
import time
from datetime import datetime, timezone
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)


class EVEAPIError(Exception):
    """Base exception for EVE API errors"""
    pass


class EVEAPITimeoutError(EVEAPIError):
    """EVE API backend timeout or network timeout"""
    pass


class EVEAPIRateLimitError(EVEAPIError):
    """EVE API rate limit exceeded"""
    pass


class EVEAPIServerError(EVEAPIError):
    """EVE API server error (5xx)"""
    pass


class EVEAPIClient:
    """EVE ESI API client with rate limiting and error handling"""

    def __init__(self):
        self.base_url = "https://esi.evetech.net/latest"
        self.headers = {
            'User-Agent': 'EVE-Market-Compression-Demo contact: fbaeckman@gmail.com',
            'Accept': 'application/json'
        }
        self.rate_limit_remaining = 100
        self.rate_limit_reset = time.time() + 60
        self.last_request_time = 0
        self.min_request_interval = 0.05  # 20 requests per second max
        self.request_timeout = 30

    def _wait_for_rate_limit(self) -> None:
        """Ensure we don't exceed rate limits"""
        current_time = time.time()

        # Wait minimum interval between requests
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)

        # Check rate limit
        if self.rate_limit_remaining <= 1:
            sleep_time = max(0, self.rate_limit_reset - current_time)
            if sleep_time > 0:
                print(f"⚠️  Rate limit reached, sleeping for {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)

        self.last_request_time = time.time()

    def _update_rate_limit_headers(self, response: httpx.Response) -> None:
        """Update rate limit tracking from response headers"""
        if 'X-ESI-Error-Limit-Remain' in response.headers:
            self.rate_limit_remaining = int(response.headers['X-ESI-Error-Limit-Remain'])

        if 'X-ESI-Error-Limit-Reset' in response.headers:
            self.rate_limit_reset = time.time() + int(response.headers['X-ESI-Error-Limit-Reset'])

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((
            httpx.TimeoutException,
            httpx.NetworkError,
            EVEAPIServerError,
            EVEAPITimeoutError
        ))
    )
    def make_request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        etag: Optional[str] = None
    ) -> httpx.Response:
        """
        Make a request to EVE ESI API with retry logic.

        Raises:
            EVEAPITimeoutError: On timeout from EVE backend
            EVEAPIRateLimitError: On rate limit (after waiting)
            EVEAPIServerError: On 5xx errors
            httpx.HTTPStatusError: On 4xx client errors (not retried)
            httpx.TimeoutException: On network timeout (retried)
        """
        self._wait_for_rate_limit()

        headers = self.headers.copy()
        if etag:
            headers['If-None-Match'] = etag

        url = f"{self.base_url}{endpoint}"

        try:
            with httpx.Client(timeout=self.request_timeout) as client:
                response = client.get(url, headers=headers, params=params or {})

                self._update_rate_limit_headers(response)

                # Handle 420 rate limit specifically
                if response.status_code == 420:
                    reset_time = int(response.headers.get('X-ESI-Error-Limit-Reset', 60))
                    print(f"⚠️  Rate limited by ESI, waiting {reset_time} seconds...")
                    time.sleep(reset_time)
                    # Retry the request recursively (within retry decorator)
                    return self.make_request(endpoint, params, etag)

                # Handle 304 Not Modified
                if response.status_code == 304:
                    return response

                # Handle 5xx server errors - raise custom exception for retry
                if response.status_code >= 500:
                    raise EVEAPIServerError(f"Server error {response.status_code}: {response.text[:200]}")

                # Raise for 4xx client errors (will NOT be retried)
                response.raise_for_status()

                # Only warn if we're getting close to rate limits
                if self.rate_limit_remaining < 20:
                    print(f"⚠️  Rate limit low: {self.rate_limit_remaining} requests remaining")

                return response

        except httpx.TimeoutException as e:
            print(f"⚠️  Request timeout - will retry: {endpoint}")
            raise EVEAPITimeoutError(f"Network timeout for {endpoint}") from e
        except httpx.HTTPStatusError as e:
            # 4xx errors - don't retry
            print(f"❌ HTTP client error: {e.response.status_code} for {endpoint}")
            raise
        except EVEAPIServerError:
            # Already logged, re-raise for retry
            raise
        except EVEAPITimeoutError:
            # Already logged, re-raise for retry
            raise
        except Exception as e:
            print(f"❌ Unexpected request failure for {endpoint}: {e}")
            raise

    def get_market_orders(
        self,
        region_id: int,
        order_type: str = "all",
        etag: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get market orders for a region"""
        all_orders = []
        page = 1
        start_time = time.time()

        print(f"📡 Fetching market orders for region {region_id}...")

        while True:
            endpoint = f"/markets/{region_id}/orders/"
            params = {
                'page': page,
                'order_type': order_type
            }

            response = self.make_request(endpoint, params, etag if page == 1 else None)

            # Handle 304 Not Modified
            if response.status_code == 304:
                return {
                    'orders': [],
                    'not_modified': True,
                    'etag': etag,
                    'last_modified': None
                }

            try:
                orders = response.json()
            except Exception as e:
                print(f"❌ Failed to parse JSON response: {e}")
                # Check if it's a timeout error from the API backend
                if response.text and "Timeout" in response.text:
                    raise EVEAPITimeoutError(f"EVE API backend timeout for region {region_id}")
                raise

            if not orders:
                break

            # Add all orders from this page
            all_orders.extend(orders)

            # Check for more pages
            total_pages = int(response.headers.get('X-Pages', 1))
            if page >= total_pages:
                break

            page += 1

            # Progress indicator
            if page % 50 == 0:
                print(f"   ... fetched {page}/{total_pages} pages ({len(all_orders):,} orders so far)")

        # Deduplicate orders by order_id (ESI API sometimes returns duplicates across pages)
        seen_order_ids = set()
        unique_orders = []
        duplicate_count = 0

        for order in all_orders:
            order_id = order.get('order_id')
            if order_id not in seen_order_ids:
                seen_order_ids.add(order_id)
                unique_orders.append(order)
            else:
                duplicate_count += 1

        if duplicate_count > 0:
            print(f"⚠️  Removed {duplicate_count} duplicate orders")

        # Parse Last-Modified header if present
        last_modified = None
        if 'Last-Modified' in response.headers:
            try:
                last_modified_str = response.headers['Last-Modified']
                # Parse Last-Modified header (RFC 7231 format: "Wed, 21 Oct 2015 07:28:00 GMT")
                last_modified = datetime.strptime(last_modified_str, '%a, %d %b %Y %H:%M:%S GMT').replace(tzinfo=timezone.utc)
            except Exception as e:
                print(f"⚠️  Failed to parse Last-Modified header: {e}")

        elapsed = time.time() - start_time
        print(f"✅ Fetched {len(unique_orders):,} orders in {elapsed:.1f}s ({page} pages)")

        return {
            'orders': unique_orders,
            'not_modified': False,
            'etag': response.headers.get('ETag'),
            'last_modified': last_modified,
            'pages_fetched': page,
            'fetch_duration': elapsed
        }

    def test_connection(self) -> bool:
        """Test API connection"""
        try:
            response = self.make_request("/status/")
            return response.status_code == 200
        except Exception as e:
            print(f"❌ API connection test failed: {e}")
            return False
