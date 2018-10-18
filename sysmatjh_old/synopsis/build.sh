if [ "$#" -ne 2 ]; then
    echo "Usage: {TEX FILENAME} {BIB FILENAME}"
    exit 1
fi

# Compile the document
pdflatex -quiet $FILENAME
if [ $? == 0 ]; then
    bibtex -quiet $FILENAME
    if [ $? == 0 ]; then
        pdflatex -quiet $FILENAME
        pdflatex -quiet $FILENAME
    fi
fi

# remove stuff
rm $FILENAME.bbl
rm $FILENAME.blg
rm $FILENAME.log
rm $FILENAME.out
rm $FILENAME.aux
