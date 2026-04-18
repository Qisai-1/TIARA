TIARA_PARENT=/work/mech-ai-scratch/supersai

case ":$PYTHONPATH:" in
    *":$TIARA_PARENT:"*) ;;
    *) export PYTHONPATH=$TIARA_PARENT:$PYTHONPATH ;;
esac

export MINARI_DATASETS_PATH=/work/mech-ai-scratch/supersai/TIARA/minari_datasets
export WANDB_PROJECT=tabrl

echo "[TabRL] Environment activated"
echo "  MINARI_DATASETS_PATH = $MINARI_DATASETS_PATH"
echo "  PYTHONPATH           = $PYTHONPATH"
