#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Running PegaFlow CI Checks ===${NC}\n"

# Check 1: Rust formatting
echo -e "${YELLOW}[1/6] Checking Rust formatting...${NC}"
cargo fmt --all -- --check
echo -e "${GREEN}✓ Formatting check passed${NC}\n"

# Check 2: Typos (optional, skip if not installed)
echo -e "${YELLOW}[2/6] Checking for typos...${NC}"
if command -v typos &> /dev/null; then
    typos
    echo -e "${GREEN}✓ Typos check passed${NC}\n"
else
    echo -e "${YELLOW}⚠ typos not installed, skipping (install with: cargo install typos-cli)${NC}\n"
fi

# Check 3: Python formatting with ruff
echo -e "${YELLOW}[3/6] Checking Python formatting with ruff...${NC}"
if ! command -v ruff &> /dev/null; then
    echo -e "${RED}✗ ruff not installed${NC}"
    echo -e "${RED}Install with: pip install ruff${NC}"
    exit 1
fi
ruff format --check python/
echo -e "${GREEN}✓ Python formatting check passed${NC}\n"

# Check 4: Python linting with ruff
echo -e "${YELLOW}[4/6] Running Python linter (ruff)...${NC}"
ruff check python/
echo -e "${GREEN}✓ Python linting passed${NC}\n"

# Check 5: Clippy for Rust packages
echo -e "${YELLOW}[5/6] Running clippy for Rust packages...${NC}"
cargo clippy -p pegaflow-core -p pegaflow-server --all-targets -- -D warnings
echo -e "${GREEN}✓ Clippy check passed${NC}\n"

# Check 6: Cargo check
echo -e "${YELLOW}[6/6] Running cargo check...${NC}"
cargo check -p pegaflow-core
cargo check -p pegaflow-server
echo -e "${GREEN}✓ Cargo check passed${NC}\n"

echo -e "${GREEN}=== All checks passed! ✓ ===${NC}"
echo -e "You can safely commit and push your changes."
