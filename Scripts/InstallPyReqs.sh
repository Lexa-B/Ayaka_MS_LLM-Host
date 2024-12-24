"$(dirname "$0")"/PyReqs/_Common.sh

echo "Install python requirements for which LLM?"
echo "1. llm-jp/llm-jp-3"
echo "2. gradientai/Llama-3-8B-Instruct-Gradient-1048k"
echo "3. deepseek/deepseek-llm-7b-base"
echo "4. meta-llama/llama-3-8b-instruct"
echo "5. aws-prototyping/MegaBeam-Mistral-7B-512k"
echo "6. mistralai/Mistral-7B-Instruct-v0.1"
read -p "Enter choice (1-5): " choice
if [ "$choice" = "1" ]; then
  echo "Installing requirements for llm-jp/llm-jp-3-13b";
  "$(dirname "$0")"/PyReqs/llm-jp-3-13b.sh
elif [ "$choice" = "2" ]; then
  echo "Installing requirements for gradientai/Llama-3-8B-Instruct-Gradient-1048k";
  "$(dirname "$0")"/PyReqs/Llama-3-8B-Instruct-Gradient-1048k.sh
elif [ "$choice" = "3" ]; then
  echo "Installing requirements for deepseek/deepseek-llm-7b-base";
  "$(dirname "$0")"/PyReqs/deepseek-llm-7b-base.sh
elif [ "$choice" = "4" ]; then
  echo "Installing requirements for meta-llama/llama-3-8b-instruct";
  "$(dirname "$0")"/PyReqs/llama-3-8b-instruct.sh
elif [ "$choice" = "5" ]; then
  echo "Installing requirements for aws-prototyping/MegaBeam-Mistral-7B-512k";
  "$(dirname "$0")"/PyReqs/MegaBeam-Mistral-7B-512k.sh
elif [ "$choice" = "6" ]; then
  echo "Installing requirements for mistralai/Mistral-7B-Instruct-v0.1";
  "$(dirname "$0")"/PyReqs/Mistral-7B-Instruct-v0.1.sh
fi
