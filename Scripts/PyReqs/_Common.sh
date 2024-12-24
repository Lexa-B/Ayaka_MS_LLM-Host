echo ""
echo "================================================================================================"
echo " --------------------------- Installing common python requirements --------------------------- "
echo "================================================================================================"
echo ""
pip install bitsandbytes==0.45.*
pip install pydantic==2.10.*
pip install transformers==4.47.*
pip install fastapi==0.115.*
pip install accelerate==1.2.*
pip install uvicorn==0.32.*
pip install torch==2.5.1+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
pip install loguru==0.7.*