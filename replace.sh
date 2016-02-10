
# Simple commands to search module files
# and replace certain commands

find . -name "*.py" -type f -exec sed -i s/fromdims/from_dims/ {} \;
