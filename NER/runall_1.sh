
echo "Run conll to onto 2"
cd ./NER/Conll2Onto_2
bash  NER_Conll2Onto_2.sh
echo "Finish conll to onto 2"
echo "----------------------"
cd ../..

echo "Run onto to conll 1"
cd ./NER/Onto2Conll_1
bash  NER_Onto2Conll_1.sh
echo "Finish onto to conll 1"
echo "----------------------"
cd ../..

echo "Run conll to onto 1"
cd ./NER/Conll2Onto_1
bash  NER_Conll2Onto_1.sh
echo "Finish conll to onto 1"
echo "----------------------"
cd ../..