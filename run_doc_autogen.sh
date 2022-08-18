sphinx-apidoc -f -o docs/source/ Nidelva3D/src/
echo "Complete auto doc generation. Remember to change index.rst and others"
cd docs/
make html
