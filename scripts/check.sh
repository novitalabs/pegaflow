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
echo -e "${YELLOW}[1/5] Checking Rust formatting...${NC}"
cargo fmt --all -- --check
echo -e "${GREEN}✓ Formatting check passed${NC}\n"

# Check 2: Typos (optional, skip if not installed)
echo -e "${YELLOW}[2/5] Checking for typos...${NC}"
if command -v typos &> /dev/null; then
    typos
    echo -e "${GREEN}✓ Typos check passed${NC}\n"
else
    echo -e "${YELLOW}⚠ typos not installed, skipping (install with: cargo install typos-cli)${NC}\n"
fi

# Check 3: Clippy for pegaflow-core
echo -e "${YELLOW}[3/5] Running clippy for pegaflow-core...${NC}"
cargo clippy -p pegaflow-core --all-targets -- -D warnings
echo -e "${GREEN}✓ Clippy check passed for pegaflow-core${NC}\n"

# Check 4: Clippy for pegaflow-server
echo -e "${YELLOW}[4/5] Running clippy for pegaflow-server...${NC}"
cargo clippy -p pegaflow-server --all-targets -- -D warnings
echo -e "${GREEN}✓ Clippy check passed for pegaflow-server${NC}\n"

# Check 5: Cargo check
echo -e "${YELLOW}[5/5] Running cargo check...${NC}"
cargo check -p pegaflow-core
cargo check -p pegaflow-server
echo -e "${GREEN}✓ Cargo check passed${NC}\n"

echo -e "${GREEN}=== All checks passed! ✓ ===${NC}"
echo -e "You can safely commit and push your changes."
