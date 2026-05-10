#!/bin/bash
# Install ponderchat as a CLI command on your system

set -e

# Find install directory
INSTALL_DIR="${HOME}/.local/bin"
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check we're in the right place
if [ ! -f "$SOURCE_DIR/ponderchat" ]; then
    echo "❌ Can't find ponderchat in $SOURCE_DIR"
    exit 1
fi

# Create install dir
mkdir -p "$INSTALL_DIR"

# Check if PATH includes install dir
if [[ ":$PATH:" != *":$INSTALL_DIR:"* ]]; then
    echo "⚠️  $INSTALL_DIR is not in your PATH"
    echo ""
    echo "Add this to your ~/.zshrc or ~/.bashrc:"
    echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
    echo ""
fi

# Create wrapper script that points to source files
cat > "$INSTALL_DIR/ponderchat" << EOF
#!/bin/bash
exec python3 "$SOURCE_DIR/ponderchat" "\$@"
EOF
chmod +x "$INSTALL_DIR/ponderchat"

echo "✓ Installed ponderchat → $INSTALL_DIR/ponderchat"
echo ""

# Check dependencies
echo "Checking dependencies..."

check_pkg() {
    if python3 -c "import $1" 2>/dev/null; then
        echo "  ✓ $1"
    else
        echo "  ✗ $1 (run: pip install $2)"
        MISSING=1
    fi
}

MISSING=0
check_pkg "anthropic" "anthropic"
check_pkg "mlx" "mlx"
check_pkg "mlx_lm" "mlx-lm"

if [ "$MISSING" = "1" ]; then
    echo ""
    echo "Install all dependencies:"
    echo "  pip install -r $SOURCE_DIR/requirements.txt"
fi

# Check API key
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo ""
    echo "⚠️  ANTHROPIC_API_KEY not set"
    echo "  Add to your ~/.zshrc:"
    echo "    export ANTHROPIC_API_KEY='your-key-here'"
fi

echo ""
echo "Try it:"
echo "  ponderchat --help"
echo "  ponderchat \"What's 2+2?\""
echo "  ponderchat                  # interactive mode"
