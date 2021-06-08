"""[Summary]

:param [ParamName]: [ParamDescription], defaults to [DefaultParamVal]
:type [ParamName]: [ParamType](, optional)
...
:raises [ErrorType]: [ErrorDescription]
...
:return: [ReturnDescription]
:rtype: [ReturnType]

Ref: https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html
"""


# reset docs
# sphinx-apidoc -o source/ ../rohan
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

echo 'cd rohan.wiki/;git commit -am "update";git push origin master'
