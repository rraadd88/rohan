# make docs
make text
# rename and copy to rohan.wiki
for f in build/text/*.txt;
do
fn=$(basename $f .txt)
# fn="${f//+(*\/|.*)}"
# fn="${f%.*}"
# fn="${f##*/}"
cp $f rohan.wiki/$fn.rst
done
# update repo
cd rohan.wiki;git status

