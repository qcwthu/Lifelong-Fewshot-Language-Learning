

echo "Run conll to onto 2"
cd /data/qin/LFPT5_code/NER/Conll2Onto_2
bash  NER_Conll2Onto_2.sh
echo "Finish conll to onto 2"
echo "----------------------"

echo "Run onto to conll 1"
cd /data/qin/LFPT5_code/NER/Onto2Conll_1
bash  NER_Onto2Conll_1.sh
echo "Finish onto to conll 1"
echo "----------------------"


echo "Run conll to onto 1"
cd /data/qin/LFPT5_code/NER/Conll2Onto_1
bash  NER_Conll2Onto_1.sh
echo "Finish conll to onto 1"
echo "----------------------"