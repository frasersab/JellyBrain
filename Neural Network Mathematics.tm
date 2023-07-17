<TeXmacs|2.1.1>

<style|generic>

<\body>
  <doc-data|<doc-title|Neural Network Mathematics>>

  <\padded-center>
    by Fraser Sabine

    version 1.0
  </padded-center>

  <\table-of-contents|toc>
    <vspace*|1fn><with|font-series|bold|math-font-series|bold|1<space|2spc>Notation>
    <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-1><vspace|0.5fn>

    <vspace*|1fn><with|font-series|bold|math-font-series|bold|2<space|2spc>Cost/Loss
    Function> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-2><vspace|0.5fn>

    <with|par-left|1tab|2.1<space|2spc>Squared error
    <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-3>>

    <with|par-left|1tab|2.2<space|2spc>Cross entropy
    <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-4>>

    <vspace*|1fn><with|font-series|bold|math-font-series|bold|3<space|2spc>Activation
    Function> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-5><vspace|0.5fn>

    <with|par-left|1tab|3.1<space|2spc>Sigmoid
    <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-6>>

    <with|par-left|1tab|3.2<space|2spc>Softmax
    <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-7>>

    <vspace*|1fn><with|font-series|bold|math-font-series|bold|4<space|2spc>Matrix
    weights> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-8><vspace|0.5fn>

    <vspace*|1fn><with|font-series|bold|math-font-series|bold|5<space|2spc>Backpropigation>
    <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-9><vspace|0.5fn>

    <with|par-left|1tab|5.1<space|2spc>Sigmoid and squared error
    <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-10>>

    <with|par-left|1tab|5.2<space|2spc>Softmax and cross entropy
    <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-11>>

    <vspace*|1fn><with|font-series|bold|math-font-series|bold|Appendix
    A<space|2spc>Cross entropy derivation>
    <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-12><vspace|0.5fn>

    <vspace*|1fn><with|font-series|bold|math-font-series|bold|Appendix
    B<space|2spc>Softmax derivation> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-13><vspace|0.5fn>
  </table-of-contents>

  <new-page*><section|Notation>

  Where <math|n> is the number of nodes in the neural network

  Where <math|L> denotes the final layer in the neural network and
  <math|L-\<ldots\>> denotes previous layers

  Where <math|<with|font-series|bold|z><rsup|L>> is the calculated output
  <math|<with|font-series|bold|z><rsup|L>=<with|font-series|bold|a><rsup|L-1>*W<rsup|L>+B<rsup|L>>\ 

  Where <math|<with|font-series|bold|a><rsup|L>> is the activation function
  applied to <math|<with|font-series|bold|z><rsup|L>>. Note
  <math|<with|font-series|bold|a><rsup|L-H> > where <math|H> is the number of
  layers, is the input layer of the neural network.

  Where <math|W<rsup|L>> is the weights matrix

  Where <math|<with|font-series|bold|b><rsup|L> is the biases matrix>

  Where <with|font-series|bold|<math|t>> is the target output given as a
  training example

  Where c is the loss/cost function

  Note <math|\<oslash\>> is the Hadamard division or the element-wise
  division of a vector or matrix

  Note <math|\<circ\>> is the Hadamard product or the element-wise
  multiplication of a vector or matrix

  Note <math|\<cdot\>> is the dot product of a vector or matrix

  <\eqnarray*>
    <tformat|<table|<row|<cell|W<rsup|L>=<matrix|<tformat|<table|<row|<cell|w<rsub|00><rsup|L>>|<cell|w<rsub|01><rsup|L>>|<cell|\<cdots\>>|<cell|w<rsub|0n><rsup|L>>>|<row|<cell|w<rsub|10><rsup|L>>|<cell|w<rsub|11><rsup|L>>|<cell|\<cdots\>>|<cell|w<rsub|1n><rsup|L>>>|<row|<cell|\<vdots\>>|<cell|\<vdots\>>|\<ddots\>|<cell|\<vdots\>>>|<row|<cell|w<rsub|n0><rsup|L>>|w<rsub|n1><rsup|L>|<cell|\<cdots\>>|<cell|w<rsub|nn><rsup|L>>>>>>>|<cell|>|<cell|<with|font-series|bold|b><rsup|L>=<matrix|<tformat|<table|<row|<cell|b<rsub|0><rsup|L>>>|<row|<cell|b<rsub|1><rsup|L>>>|<row|<cell|\<vdots\>>>|<row|<cell|b<rsub|n><rsup|L>>>>>>>>>>
  </eqnarray*>

  <\eqnarray*>
    <tformat|<table|<row|<cell|>|<cell|<with|font-series|bold|z<rsup|<with|font-series||L>>>=W<rsup|L>\<cdot\><with|font-series|bold|a><rsup|L-1>+<with|font-series|bold|b><rsup|L>=<matrix|<tformat|<table|<row|<cell|*w<rsub|00><rsup|L>*a<rsub|0><rsup|L-1>+*w<rsub|01><rsup|L>*a<rsub|1><rsup|L-1>+\<cdots\>+w<rsub|0n><rsup|L>*a<rsub|n><rsup|L-1>+b<rsub|0><rsup|L>>>|<row|<cell|*w<rsub|10><rsup|L>*a<rsub|1><rsup|L-1>+*w<rsub|11><rsup|L>*a<rsub|1><rsup|L-1>+\<cdots\>+*w<rsub|1n><rsup|L>*a<rsub|n><rsup|L-1>+b<rsub|1><rsup|L>>>|<row|<cell|\<vdots\>>>|<row|<cell|*w<rsub|n0><rsup|L>*a<rsub|n><rsup|L-1>+w<rsub|n1><rsup|L>*a<rsub|n><rsup|L-1>*+\<cdots\>+w<rsub|nn><rsup|L>*a<rsub|n><rsup|L-1>*+b<rsub|n><rsup|L>>>>>>=<matrix|<tformat|<table|<row|<cell|z<rsub|0>>>|<row|<cell|z<rsub|1>>>|<row|<cell|\<vdots\>>>|<row|<cell|z<rsub|n>>>>>>>|<cell|>>|<row|<cell|>|<cell|z<rsub|j><rsup|L>=<big|sum><rsub|k=0><rsup|n>w<rsub|jk><rsup|L>*a<rsub|k><rsup|L-1><rsub|>+b<rsub|j><rsup|L>>|<cell|>>>>
  </eqnarray*>

  <\padded-center>
    where f() is the activation function e.g. sigmoid, tanh, relu etc\ 
  </padded-center>

  <\eqnarray*>
    <tformat|<table|<row|<cell|>|<cell|<with|font-series|bold|a><rsup|L>=<matrix|<tformat|<table|<row|<cell|f<around*|(|z<rsub|0>|)>>>|<row|<cell|f<around*|(|z<rsub|1>|)>>>|<row|<cell|\<vdots\>>>|<row|<cell|f<around*|(|z<rsub|n>|)>>>>>>=<matrix|<tformat|<table|<row|<cell|a<rsup|L><rsub|0>>>|<row|<cell|a<rsup|L><rsub|1>>>|<row|<cell|\<vdots\>>>|<row|<cell|a<rsup|L><rsub|n>>>>>>>|<cell|>>>>
  </eqnarray*>

  <new-page*><section|Loss/Cost Function>

  The loss function is generally for a single training example whereas the
  cost function is usually the mean of the loss function over a batch (or
  mini-batch) of training examples. Doing gradient decent with batches (or
  mini-batches) is more computationally efficient as backpropigation is
  calculated only as many times as there are batches. The loss/cost function
  needs to produce a scalar value to make gradient decent possible.

  <subsection|Squared error>

  Note 1: this is very similar to the mean squared error function however the
  <math|<tfrac|1|n>> is omitted as it is quicker to compute without it and
  doesn't meaningfully affect the gradient descent.

  Note 2: <math|<tfrac|1|2>> is multiplied as a constant to the general
  squared error function to remove the constant from the derivative.\ 

  <\padded-center>
    Element form
  </padded-center>

  <\eqnarray*>
    <tformat|<table|<row|<cell|>|<cell|c==<dfrac|1|2>*<big|sum><rsup|n><rsub|j=0><around*|(|<math-it|<with|font-series||t>><rsub|j>-<with|font-series||a><rsub|j><rsup|L>|)><rsup|2>=<dfrac|1|2><around*|(|<around*|(|<math-it|<with|font-series|bold|t>><rsub|0>-<with|font-series|bold|a><rsub|0><rsup|L>|)><rsup|2>+<around*|(|<math-it|<with|font-series|bold|t>><rsub|1>-<with|font-series|bold|a><rsub|1><rsup|L>|)><rsup|2>+\<cdots\>+<around*|(|<math-it|<with|font-series|bold|t>><rsub|n>-<with|font-series|bold|a><rsub|n><rsup|L>|)><rsup|2>|)>*>|<cell|>>|<row|<cell|>|<cell|<dfrac|\<mathd\><math-it|<with|font-series||c>><with|font-series|bold|>|\<mathd\><with|font-series||a><rsub|j><rsup|L>>=t<rsub|j>-a<rsup|L><rsub|j>>|<cell|>>>>
  </eqnarray*>

  \;

  <\padded-center>
    Matrix Form
  </padded-center>

  <\eqnarray*>
    <tformat|<table|<row|<cell|>|<cell|<with|font-series||<math-it|<with|font-series||c>>>=<dfrac|1|2>***<around*|(|<around*|(|<math-it|<with|font-series|bold|t>><rsub|>-<with|font-series|bold|a><rsub|><rsup|L>|)>\<circ\><around*|(|<math-it|<with|font-series|bold|t>><rsub|>-<with|font-series|bold|a><rsub|><rsup|L>|)><rsup|>|)>>|<cell|>>|<row|<cell|>|<cell|<dfrac|\<mathd\><math-it|<with|font-series||c>><with|font-series|bold|>|\<mathd\><with|font-series|bold|a><rsup|L>>=*<math-it|<with|font-series|bold|t>>-<with|font-series|bold|a><rsup|L>=<matrix|<tformat|<table|<row|<cell|<math-it|<with|font-series||t>><rsub|0>-<with|font-series||a><rsub|0><rsup|L>>>|<row|<cell|<math-it|<with|font-series||t>><rsub|1>-<with|font-series||a><rsub|1><rsup|L>>>|<row|<cell|\<vdots\>>>|<row|<cell|<math-it|<with|font-series||t>><rsub|n>-<with|font-series||a><rsub|n><rsup|L>>>>>>>|<cell|>>>>
  </eqnarray*>

  <subsection|Cross entropy>

  <em|<\padded-center>
    <em|Element form>
  </padded-center>>

  <\eqnarray*>
    <tformat|<table|<row|<cell|>|<cell|<math-it|<with|font-series||c>>=-<big|sum><rsub|j=0><rsup|n><with|font-series||t<rsub|j>*log<rsub|b><with|font-series|bold|<around*|(|<with|font-series||a<rsup|L><rsub|j>>|)>>>=-<around*|(|t<rsub|0>*log<rsub|b><around*|(|<with|font-series||a<rsup|L><rsub|0>>|)><with|font-series|bold|>+t<rsub|1>*log<rsub|b><with|font-series|bold|<around*|(|<with|font-series||a<rsup|L><rsub|1>>|)>>+\<cdots\>+t<rsub|n>*log<rsub|b><with|font-series|bold|<around*|(|<with|font-series||a<rsup|L><rsub|n>>|)>>|)>>|<cell|>>|<row|<cell|>|<cell|<dfrac|\<mathd\><with|font-series||c>|\<mathd\><with|font-series||a<rsup|L><rsub|j>>>=-<dfrac|1|ln<around*|(|b|)>>*<dfrac|t<rsub|j>|a<rsup|L><rsub|j>*>>|<cell|>>>>
  </eqnarray*>

  \;

  <\padded-center>
    Matrix form
  </padded-center>

  <\eqnarray*>
    <tformat|<table|<row|<cell|>|<cell|c=-<with|font-series|bold|t><rsup|T>\<cdot\>log<rsub|b><around*|(|<with|font-series||<with|font-series|bold|a><rsup|L>>|)>>|<cell|>>|<row|<cell|>|<cell|<dfrac|\<mathd\><with|font-series||c>|\<mathd\><with|font-series|bold|a><rsup|L>>=-<dfrac|1|ln<around*|(|b|)>>*<around*|(|<with|font-series|bold|t>\<oslash\><with|font-series|bold|a><rsup|L>|)>=-<dfrac|1|ln<around*|(|b|)>>*<matrix|<tformat|<table|<row|<cell|<dfrac|t<rsub|0>|a<rsup|L><rsub|0>*>>>|<row|<cell|<dfrac|t<rsub|1>|a<rsup|L><rsub|1>*>>>|<row|<cell|\<vdots\>>>|<row|<cell|<dfrac|t<rsub|n>|a<rsup|L><rsub|n>>>>>>>>|<cell|>>>>
  </eqnarray*>

  Please see appendix A for full cross entropy derivation.

  <new-page*><section|Activation Function>

  Note that the backpropigation is meaningfully different if the activation
  function is a vector or scalar function.

  <subsection|Sigmoid>

  <\padded-center>
    Element form
  </padded-center>

  <\eqnarray*>
    <tformat|<table|<row|<cell|>|<cell|a<rsub|j><rsup|L>=\<sigma\><around*|(|<with|font-series||z><rsub|j><rsup|L>|)>=<dfrac|1|1+e<rsup|-<with|font-series||z<rsub|j><rsup|>><rsup|L>>>>|<cell|>>|<row|<cell|>|<cell|<dfrac|\<mathd\><with|font-series||a><rsub|j><rsup|L>|\<mathd\><with|font-series||z><rsub|j><rsup|L>>=<dfrac|\<mathd\>\<sigma\><around*|(|<with|font-series||z><rsup|L>|)>|\<mathd\><with|font-series||z><rsup|L>>=\<sigma\><around*|(|<with|font-series||z><rsub|j><rsup|L>|)>*<around*|(|1-\<sigma\><around*|(|<with|font-series||z><rsub|j><rsup|L>|)>|)>=<with|font-series||a><rsub|j><rsup|L>*<around*|(|1-<with|font-series||a><rsub|j><rsup|L>|)>>|<cell|>>>>
  </eqnarray*>

  \;

  <\padded-center>
    Matrix Form
  </padded-center>

  <\eqnarray*>
    <tformat|<table|<row|<cell|>|<cell|<with|font-series|bold|a><rsup|L>=\<sigma\><around*|(|<with|font-series|bold|z><rsup|L>|)>=<dfrac|1|1+e<rsup|-<with|font-series|bold|z<rsup|>><rsup|L>>>>|<cell|>>|<row|<cell|>|<cell|<dfrac|\<mathd\><with|font-series|bold|a><rsup|L>|\<mathd\><with|font-series|bold|z><rsup|L>>=<dfrac|\<mathd\>\<sigma\><around*|(|<with|font-series|bold|z><rsup|L>|)>|\<mathd\><with|font-series|bold|z><rsup|L>>=\<sigma\><around*|(|<with|font-series|bold|z><rsup|L>|)>\<circ\><around*|(|1-\<sigma\><around*|(|<with|font-series|bold|z><rsup|L>|)>|)>=<with|font-series|bold|a><rsup|L>\<circ\><around*|(|1-<with|font-series|bold|a><rsup|L>|)>>|<cell|>>>>
  </eqnarray*>

  <subsection|Softmax>

  <\padded-center>
    Element form
  </padded-center>

  <\eqnarray*>
    <tformat|<table|<row|<cell|>|<cell|a<rsub|j><rsup|L>=<dfrac|e<rsup|<with|font-series||z><rsub|j><rsup|L>>|<big|sum><rsup|n><rsub|i=0><rsub|>e<rsup|<with|font-series|bold|<with|font-series||z<rsup|L><rsub|i>>><rsup|>>>>|<cell|>>|<row|<cell|>|<cell|<text|where
    <math|\<delta\>> is the Kronecker delta: <math|\<delta\><rsub|j
    k><choice|<tformat|<table|<row|<cell|1 if j=k>>|<row|<cell|0 if
    j\<neq\>k>>>>>>>>|<cell|>>|<row|<cell|>|<cell|<dfrac|\<mathd\><with|font-series||a><rsub|j><rsup|L>|\<mathd\><with|font-series||z<rsub|k>><rsup|L>>=a<rsup|L><rsub|j>*<around*|(|\<delta\><rsub|j
    k>-a<rsup|L><rsub|k>|)>>|<cell|>>>>
  </eqnarray*>

  \;

  <\padded-center>
    Matrix form
  </padded-center>

  <\eqnarray*>
    <tformat|<table|<row|<cell|>|<cell|<with|font-series|bold|a><rsup|L>=S<around*|(|<with|font-series|bold|z><rsup|L>|)>=<dfrac|e<rsup|<with|font-series|bold|z><rsup|L>>|<big|sum><rsup|n><rsub|i=0><rsub|>e<rsup|<with|font-series|bold|<with|font-series||z<rsup|L><rsub|i>>><rsup|>>>=<dfrac|e<rsup|<with|font-series|bold|z<rsup|L><rsub|>><rsup|>>|e<rsup|<with|font-series||<with|font-series|bold|z><rsup|L><rsub|0>><rsup|>>+e<rsup|<with|font-series||<with|font-series|bold|z><rsup|L><rsub|1>><rsup|>>+\<cdots\>+e<rsup|<with|font-series||<with|font-series|bold|z><rsub|n>><rsup|L>>>>|<cell|>>|<row|<cell|>|<cell|<dfrac|\<mathd\><with|font-series|bold|a><rsup|L>|\<mathd\><with|font-series|bold|z><rsup|L>>=<dfrac|\<mathd\>S<around*|(|<with|font-series|bold|z><rsup|L>|)><rsup|>|\<mathd\><with|font-series|bold|z><rsup|L>>=<matrix|<tformat|<table|<row|<cell|a<rsup|L><rsub|0>*<around*|(|1-a<rsup|L><rsub|0>|)>>|<cell|-a<rsup|L><rsub|0>*a<rsup|L><rsub|1>>|<cell|\<cdots\>>|<cell|-a<rsup|L><rsub|0>*a<rsup|L><rsub|n>>>|<row|<cell|-a<rsup|L><rsub|1>*a<rsup|L><rsub|0>>|<cell|a<rsup|L><rsub|1
    >*<around*|(|1-a<rsup|L><rsub|1>|)>>|<cell|\<cdots\>>|<cell|-a<rsup|L><rsub|1>*a<rsup|L><rsub|n>>>|<row|<cell|\<vdots\>>|<cell|\<vdots\>>|<cell|\<ddots\>>|<cell|\<vdots\>>>|<row|<cell|-a<rsup|L><rsub|n>*a<rsup|L><rsub|0>>|<cell|-a<rsup|L><rsub|n>*a<rsup|L><rsub|1>>|<cell|\<cdots\>>|<cell|a<rsup|L><rsub|n
    >*<around*|(|1-a<rsup|L><rsub|n>|)>>>>>>>|<cell|>>>>
  </eqnarray*>

  Please see appendix B for full softmax derivation.

  <new-page*><section|Matrix weights>

  <\padded-center>
    Element form
  </padded-center>

  <\eqnarray*>
    <tformat|<table|<row|<cell|>|<cell|<with|font-series||z><rsub|j><rsup|L>=<with|font-series||a><rsub|k><rsup|L-1>*w<rsub|jk><rsup|L>+<with|font-series||b><rsub|j><rsup|L>>|<cell|>>|<row|<cell|>|<cell|<dfrac|\<mathd\><with|font-series||z<rsub|j>><rsup|L>|\<mathd\>w<rsub|j
    k><rsup|L>>=<with|font-series||a><rsub|k><rsup|L-1>>|<cell|>>|<row|<cell|>|<cell|<dfrac|\<mathd\><with|font-series||z<rsub|j>><rsup|L>|\<mathd\>b<rsub|j><rsup|L>>=1>|<cell|>>|<row|<cell|>|<cell|<dfrac|\<mathd\><with|font-series||z<rsub|j>><rsup|L>|\<mathd\><with|font-series||a><rsub|k><rsup|L-1>>=w<rsub|j
    k><rsup|L>>|<cell|>>>>
  </eqnarray*>

  \;

  <\padded-center>
    Matrix form
  </padded-center>

  <\eqnarray*>
    <tformat|<table|<row|<cell|>|<cell|<with|font-series|bold|z><rsup|L>=W<rsup|L>\<cdot\><with|font-series|bold|a><rsup|L-1>+<with|font-series|bold|b><rsup|L>>|<cell|>>|<row|<cell|>|<cell|=<matrix|<tformat|<table|<row|<cell|w<rsub|00>>|<cell|w<rsub|01>>|<cell|\<cdots\>>|<cell|w<rsub|0n>>>|<row|<cell|w<rsub|10>>|<cell|w<rsub|11>>|<cell|\<cdots\>>|<cell|w<rsub|1n>>>|<row|<cell|\<vdots\>>|<cell|\<vdots\>>|<cell|\<ddots\>>|<cell|\<vdots\>>>|<row|<cell|w<rsub|n0>>|<cell|w<rsub|n1>>|<cell|\<cdots\>>|<cell|w<rsub|nn>>>>>><matrix|<tformat|<table|<row|<cell|a<rsup|L-1><rsub|0>>>|<row|<cell|a<rsup|L-1><rsub|1>>>|<row|<cell|\<vdots\>>>|<row|<cell|a<rsup|L-1><rsub|n>>>>>>+<matrix|<tformat|<table|<row|<cell|b<rsub|0><rsup|L>>>|<row|<cell|b<rsub|1><rsup|L>>>|<row|<cell|\<vdots\>>>|<row|<cell|b<rsub|n><rsup|L>>>>>>>|<cell|>>|<row|<cell|>|<cell|<dfrac|\<mathd\><with|font-series|bold|z><rsup|L>|\<mathd\>W<rsup|L>>=<with|font-series|bold|a><rsup|L-1>>|<cell|>>|<row|<cell|>|<cell|<dfrac|\<mathd\><with|font-series|bold|z><rsup|L>|\<mathd\><with|font-series|bold|b><rsup|L>>=1>|<cell|>>|<row|<cell|>|<cell|<dfrac|\<mathd\><with|font-series|bold|z><rsup|L>|\<mathd\><with|font-series|bold|a><rsup|L-1>>=W<rsup|L>>|<cell|>>>>
  </eqnarray*>

  <new-page*><section|Backpropigation>

  <subsection|Sigmoid and squared error>

  <subsubsection|Element form>

  Layer <math|L>

  <\eqnarray*>
    <tformat|<table|<row|<cell|>|<cell|<dfrac|\<mathd\><with|font-series||c>|\<mathd\>w<rsub|jk><rsup|L>>=<dfrac|\<mathd\><with|font-series||c>|\<mathd\><with|font-series||a<rsub|j>><rsup|L>>*<dfrac|\<mathd\><with|font-series||a<rsub|j>><rsup|L>|\<mathd\><with|font-series||z<rsub|j>><rsup|L>>*<dfrac|\<mathd\><with|font-series||z<rsub|j>><rsup|L>|*\<mathd\>w<rsub|jk><rsup|L>>>|<cell|>>|<row|<cell|>|<cell|<dfrac|\<mathd\><with|font-series||c>|\<mathd\><with|font-series||a<rsub|j>><rsup|L>>=t<rsub|j>-a<rsup|L><rsub|j>>|<cell|>>|<row|<cell|>|<cell|<dfrac|\<mathd\><with|font-series||a<rsub|j>><rsup|L>|\<mathd\><with|font-series||z<rsub|j>><rsup|L>>=<with|font-series||a><rsub|j><rsup|L>*<around*|(|1-<with|font-series||a><rsub|j><rsup|L>|)>>|<cell|>>|<row|<cell|>|<cell|<dfrac|\<mathd\><with|font-series||c>|\<mathd\><with|font-series||<with|font-series||z<rsub|j>><rsup|L>><rsup|>>=<dfrac|\<mathd\><with|font-series||c>|\<mathd\><with|font-series||a<rsub|j>><rsup|L>>*<dfrac|\<mathd\><with|font-series||a<rsub|j>><rsup|L>|\<mathd\><with|font-series||z<rsub|j>><rsup|L>>=<around*|(|t<rsub|j>-a<rsup|L><rsub|j>|)>*<with|font-series||a><rsub|j><rsup|L>*<around*|(|1-<with|font-series||a><rsub|j><rsup|L>|)>=<around*|(|t<rsub|j>-a<rsup|L><rsub|j>|)><with|font-series||*>*<around*|(|1-<with|font-series||a><rsub|j><rsup|L>|)>*<with|font-series||a><rsub|j><rsup|L>>|<cell|>>|<row|<cell|>|<cell|<dfrac|\<mathd\><with|font-series||z<rsub|j>><rsup|L>|\<mathd\>w<rsub|jk><rsup|L>>=<with|font-series||a><rsub|k><rsup|L-1>>|<cell|>>|<row|<cell|>|<cell|<dfrac|\<mathd\><with|font-series||c>|\<mathd\>w<rsub|jk><rsup|L>>=<dfrac|\<mathd\><with|font-series||c>|\<mathd\><with|font-series||<with|font-series||z<rsub|j>><rsup|L>><rsup|>>*<dfrac|\<mathd\><with|font-series||z<rsub|j>><rsup|L>|\<mathd\>w<rsub|jk><rsup|L>>=<around*|(|t<rsub|j>-a<rsup|L><rsub|j>|)><with|font-series||*>*<around*|(|1-<with|font-series||a><rsub|j><rsup|L>|)>*<with|font-series||a><rsub|j><rsup|L>**<with|font-series||a><rsub|k><rsup|L-1>>|<cell|>>>>
  </eqnarray*>

  Layer <math|L-1>

  <\eqnarray*>
    <tformat|<table|<row|<cell|>|<cell|<dfrac|\<mathd\><with|font-series||c>|\<mathd\>w<rsub|k
    q><rsup|L-1>>=<around*|(|<big|sum><rsup|n><rsub|j=0><dfrac|\<mathd\><with|font-series||c>|\<mathd\><with|font-series||<with|font-series||z<rsub|j>><rsup|L>><rsup|>>*<dfrac|\<mathd\><with|font-series||z<rsub|j>><rsup|L>|*\<mathd\>a<rsub|k><rsup|L-1>>|)>*<dfrac|\<mathd\>a<rsub|k><rsup|L-1>|\<mathd\>z<rsub|k><rsup|L-1>>*<dfrac|\<mathd\>z<rsub|k><rsup|L-1>|\<mathd\>w<rsub|k
    q><rsup|L-1>>>|<cell|>>|<row|<cell|>|<cell|<dfrac|\<mathd\><with|font-series||z<rsub|j>><rsup|L>|*\<mathd\>a<rsub|k><rsup|L-1>>=w<rsub|j
    k><rsup|L>>|<cell|>>|<row|<cell|>|<cell|<dfrac|\<mathd\><with|font-series||c>|\<mathd\>a<rsub|k><rsup|L-1>>=<big|sum><rsup|n><rsub|j=0><dfrac|\<mathd\><with|font-series||c>|\<mathd\><with|font-series||<with|font-series||z<rsub|j>><rsup|L>><rsup|>>*<dfrac|\<mathd\><with|font-series||z<rsub|j>><rsup|L>|*\<mathd\>a<rsub|k><rsup|L-1>>=<big|sum><rsup|n><rsub|j=0><around*|(|t<rsub|j>-a<rsup|L><rsub|j>|)><with|font-series||*>*<around*|(|1-<with|font-series||a><rsub|j><rsup|L>|)>*<with|font-series||a><rsub|j><rsup|L>*w<rsub|j
    k><rsup|L>>|<cell|>>|<row|<cell|>|<cell|<dfrac|\<mathd\>a<rsub|k><rsup|L-1>|\<mathd\>z<rsub|k><rsup|L-1>>=<with|font-series||a><rsub|k><rsup|L-1>*<around*|(|1-<with|font-series||a><rsub|k><rsup|L-1>|)>>|<cell|>>|<row|<cell|>|<cell|<dfrac|\<mathd\>z<rsub|k><rsup|L-1>|\<mathd\>w<rsub|k
    q><rsup|L-1>>=<with|font-series||a><rsub|q><rsup|L-1>>|<cell|>>|<row|<cell|>|<cell|<dfrac|\<mathd\><with|font-series||c>|\<mathd\>w<rsub|k
    q><rsup|L-1>>=<dfrac|\<mathd\><with|font-series||c>|\<mathd\>a<rsub|k><rsup|L-1>>*<dfrac|\<mathd\>a<rsub|k><rsup|L-1>|\<mathd\>z<rsub|k><rsup|L-1>>*<dfrac|\<mathd\>z<rsub|k><rsup|L-1>|\<mathd\>w<rsub|k
    q><rsup|L-1>>=<around*|(|<big|sum><rsup|n><rsub|j=0><around*|(|t<rsub|j>-a<rsup|L><rsub|j>|)><with|font-series||*>*<around*|(|1-<with|font-series||a><rsub|j><rsup|L>|)>*<with|font-series||a><rsub|j><rsup|L>*w<rsub|j
    k><rsup|L>|)>*<around*|(|<with|font-series||a><rsub|k><rsup|L-1>*<around*|(|1-<with|font-series||a><rsub|k><rsup|L-1>|)>|)>*<around*|(|<with|font-series||a><rsub|q><rsup|L-1>|)>>|<cell|>>>>
  </eqnarray*>

  \;

  \;

  <new-page*><subsubsection|Matrix form>

  Layer <math|L>

  <\eqnarray*>
    <tformat|<table|<row|<cell|>|<cell|<dfrac|\<mathd\><with|font-series||c>|\<mathd\>W<rsup|L>>=<around*|(|<dfrac|\<mathd\><with|font-series||c>|\<mathd\><with|font-series||<with|font-series|bold|a>><rsup|L>>\<circ\><dfrac|\<mathd\><with|font-series|bold|a><rsup|L>|\<mathd\><with|font-series||<with|font-series|bold|z>><rsup|L>>|)>\<cdot\><dfrac|\<mathd\><with|font-series|bold|z><rsup|L>|*\<mathd\>W<rsup|L><rsup|>><rsup|T>>|<cell|>>|<row|<cell|>|<cell|<around*|(|<matrix|<tformat|<table|<row|<cell|t<rsub|0>-a<rsup|L><rsub|0>>>|<row|<cell|t<rsub|1>-a<rsup|L><rsub|1>>>|<row|<cell|\<vdots\>>>|<row|<cell|t<rsub|n>-a<rsup|L><rsub|n>>>>>>\<circ\><matrix|<tformat|<table|<row|<with|font-series||a><rsub|0><rsup|L>*<around*|(|1-<with|font-series||a><rsub|0><rsup|L>|)>>|<row|<cell|<with|font-series||a><rsub|1><rsup|L>*<around*|(|1-<with|font-series||a><rsub|1><rsup|L>|)>>>|<row|<cell|\<vdots\>>>|<row|<cell|<with|font-series||a><rsub|n><rsup|L>*<around*|(|1-<with|font-series||a><rsub|n><rsup|L>|)>>>>>>|)>\<cdot\><matrix|<tformat|<table|<row|<cell|<with|font-series||a><rsub|0><rsup|L-1>>>|<row|<cell|<with|font-series||a><rsub|1><rsup|L-1>>>|<row|<cell|\<vdots\>>>|<row|<cell|<with|font-series||a><rsub|n><rsup|L-1>>>>>><rsup|T>>|<cell|>>|<row|<cell|>|<cell|=<matrix|<tformat|<table|<row|<cell|<around*|(|t<rsub|0>-a<rsup|L><rsub|0>|)><with|font-series||*>*<around*|(|1-<with|font-series||a><rsub|0><rsup|L>|)>*<with|font-series||a><rsub|0><rsup|L>**<with|font-series||a><rsub|0><rsup|L-1>>|<cell|<around*|(|t<rsub|0>-a<rsup|L><rsub|0>|)><with|font-series||*>*<around*|(|1-<with|font-series||a><rsub|0><rsup|L>|)>*<with|font-series||a><rsub|0><rsup|L>**<with|font-series||a><rsub|1><rsup|L-1>>|<cell|\<cdots\>>|<cell|<around*|(|t<rsub|0>-a<rsup|L><rsub|0>|)><with|font-series||*>*<around*|(|1-<with|font-series||a><rsub|0><rsup|L>|)>*<with|font-series||a><rsub|0><rsup|L>**<with|font-series||a><rsub|n><rsup|L-1>>>|<row|<cell|<around*|(|t<rsub|1>-a<rsup|L><rsub|1>|)><with|font-series||*>*<around*|(|1-<with|font-series||a><rsub|1><rsup|L>|)>*<with|font-series||a><rsub|1><rsup|L>**<with|font-series||a><rsub|0><rsup|L-1>>|<cell|<around*|(|t<rsub|1>-a<rsup|L><rsub|1>|)><with|font-series||*>*<around*|(|1-<with|font-series||a><rsub|1><rsup|L>|)>*<with|font-series||a><rsub|1><rsup|L>**<with|font-series||a><rsub|1><rsup|L-1>>|<cell|\<cdots\>>|<cell|<around*|(|t<rsub|1>-a<rsup|L><rsub|1>|)><with|font-series||*>*<around*|(|1-<with|font-series||a><rsub|1><rsup|L>|)>*<with|font-series||a><rsub|1><rsup|L>**<with|font-series||a><rsub|n><rsup|L-1>>>|<row|<cell|\<vdots\>>|<cell|\<vdots\>>|<cell|\<ddots\>>|<cell|\<vdots\>>>|<row|<cell|<around*|(|t<rsub|n>-a<rsup|L><rsub|n>|)><with|font-series||*>*<around*|(|1-<with|font-series||a><rsub|n><rsup|L>|)>*<with|font-series||a><rsub|n><rsup|L>**<with|font-series||a><rsub|0><rsup|L-1>>|<cell|<around*|(|t<rsub|n>-a<rsup|L><rsub|n>|)><with|font-series||*>*<around*|(|1-<with|font-series||a><rsub|n><rsup|L>|)>*<with|font-series||a><rsub|n><rsup|L>**<with|font-series||a><rsub|1><rsup|L-1>>|<cell|\<cdots\>>|<cell|<around*|(|t<rsub|n>-a<rsup|L><rsub|n>|)><with|font-series||*>*<around*|(|1-<with|font-series||a><rsub|n><rsup|L>|)>*<with|font-series||a><rsub|n><rsup|L>**<with|font-series||a><rsub|n><rsup|L-1>>>>>>>|<cell|>>>>
  </eqnarray*>

  Layer <math|L-1>

  <\eqnarray*>
    <tformat|<table|<row|<cell|>|<cell|<dfrac|\<mathd\><with|font-series||c>|\<mathd\>W<rsup|L-1>>=<around*|(|<around*|(|<dfrac|\<mathd\><with|font-series||c>|\<mathd\><with|font-series||<with|font-series||z><rsup|L>><rsup|>><rsup|T>*\<cdot\><dfrac|\<mathd\><with|font-series||z><rsup|L>|*\<mathd\>a<rsup|L-1>>|)>\<circ\><dfrac|\<mathd\>a<rsup|L-1>|\<mathd\>z<rsup|L-1>>|)>\<cdot\><dfrac|\<mathd\>z<rsup|L-1>|\<mathd\>w<rsup|L-1>>>|<cell|>>|<row|<cell|>|<cell|<dfrac|\<mathd\><with|font-series||z><rsup|L>|*\<mathd\>a<rsup|L-1>>=W<rsup|L>>|<cell|>>|<row|<cell|>|<cell|<dfrac|\<mathd\>a<rsup|L-1>|\<mathd\>z<rsup|L-1>>=<matrix|<tformat|<table|<row|<with|font-series||a><rsub|0><rsup|L-1>*<around*|(|1-<with|font-series||a><rsub|0><rsup|L-1>|)>>|<row|<cell|<with|font-series||a><rsub|1><rsup|L-1>*<around*|(|1-<with|font-series||a><rsub|1><rsup|L-1>|)>>>|<row|<cell|\<vdots\>>>|<row|<cell|<with|font-series||a><rsub|n><rsup|L-1>*<around*|(|1-<with|font-series||a><rsub|n><rsup|L-1>|)>>>>>>>|<cell|>>|<row|<cell|>|<cell|<dfrac|\<mathd\>z<rsup|L-1>|\<mathd\>w<rsup|L-1>>=<with|font-series||a><rsup|L-2>>|<cell|>>|<row|<cell|>|<cell|>|<cell|>>>>
  </eqnarray*>

  <new-page*><subsection|Softmax and cross entropy>

  <subsubsection|Element form>

  On the left <math|<with|font-series|bold|a><rsup|L>> is left in vector form
  and the dot product is taken. Instead on the right the summation of the
  elements is calculated but they are equivalent.

  <\eqnarray*>
    <tformat|<table|<row|<cell|<dfrac|\<mathd\>c|\<mathd\>w<rsub|jk><rsup|L>>=<around*|(|<dfrac|\<mathd\>c|\<mathd\><with|font-series|bold|a><rsup|L>><rsup|T>*\<cdot\><dfrac|\<mathd\><with|font-series|bold|a><rsup|L>|\<mathd\>z<rsub|j><rsup|L>>|)>**<dfrac|\<mathd\>z<rsub|j><rsub|><rsup|L>|\<mathd\>w<rsub|jk><rsup|L>>>|<cell|<text|<space|4em>>>|<cell|<dfrac|\<mathd\>c|\<mathd\>w<rsub|jk><rsup|L>>=<around*|(|<big|sum><rsub|i=0><rsup|n><dfrac|\<mathd\>c|\<mathd\><with|font-series||a><rsub|i><rsup|L>>*<dfrac|\<mathd\><with|font-series||a<rsub|i>><rsup|L>|\<mathd\>z<rsub|j><rsup|L>>|)>**<dfrac|\<mathd\>z<rsub|j><rsub|><rsup|L>|\<mathd\>w<rsub|jk><rsup|L>>>>|<row|<cell|<dfrac|\<mathd\>c|\<mathd\><with|font-series|bold|a><rsup|L>>=-<dfrac|1|ln<around*|(|b|)>>*<matrix|<tformat|<table|<row|<cell|<dfrac|t<rsub|0>|a<rsup|L><rsub|0>*>>>|<row|<cell|<dfrac|t<rsub|1>|a<rsup|L><rsub|1>*>>>|<row|<cell|\<vdots\>>>|<row|<cell|<dfrac|t<rsub|n>|a<rsup|L><rsub|n>>>>>>>>|<cell|>|<cell|<dfrac|\<mathd\>c|\<mathd\>a<rsub|i><rsup|L>>=-<dfrac|1|ln<around*|(|b|)>>**<dfrac|t<rsub|i>|a<rsup|L><rsub|i>*>>>|<row|<cell|<dfrac|\<mathd\><with|font-series|bold|a><rsup|L>|\<mathd\>z<rsub|j><rsup|L>>=<matrix|<tformat|<table|<row|<cell|a<rsup|L><rsub|0>*<around*|(|\<delta\><rsub|0j>-a<rsup|L><rsub|j>|)>>>|<row|<cell|a<rsup|L><rsub|1>*<around*|(|\<delta\><rsub|1j>-a<rsup|L><rsub|j>|)>>>|<row|<cell|\<vdots\>>>|<row|<cell|a<rsup|L><rsub|n>*<around*|(|\<delta\><rsub|n
    j>-a<rsup|L><rsub|j>|)>>>>>>>|<cell|>|<cell|<dfrac|\<mathd\>a<rsub|i><rsup|L>|\<mathd\>z<rsub|j><rsup|L>>=a<rsup|L><rsub|i>*<around*|(|\<delta\><rsub|ij>-a<rsup|L><rsub|j>|)>>>|<row|<cell|<dfrac|\<mathd\>c|\<mathd\>z<rsub|j><rsup|L>>=<dfrac|\<mathd\>c|\<mathd\><with|font-series|bold|a><rsup|L>><rsup|T>*\<cdot\><dfrac|\<mathd\><with|font-series|bold|a><rsup|L>|\<mathd\>z<rsub|j><rsup|L>>>|<cell|>|<cell|<dfrac|\<mathd\>c|\<mathd\>z<rsub|j><rsup|L>>=<big|sum><rsub|i=0><rsup|n><dfrac|\<mathd\>c|\<mathd\><with|font-series||a><rsub|i><rsup|L>>*<dfrac|\<mathd\><with|font-series||a<rsub|i>><rsup|L>|\<mathd\>z<rsub|j><rsup|L>>>>>>
  </eqnarray*>

  The multiplication of both forms is too long to write side by side

  <\eqnarray*>
    <tformat|<table|<row|<cell|>|<cell|<dfrac|\<mathd\>c|\<mathd\>z<rsub|j><rsup|L>>=<big|sum><rsub|i=0><rsup|n><dfrac|\<mathd\>c|\<mathd\><with|font-series||a><rsub|i><rsup|L>>*<dfrac|\<mathd\><with|font-series||a<rsub|i>><rsup|L>|\<mathd\>z<rsub|j><rsup|L>>=<big|sum><rsub|i=0><rsup|n><around*|(|-<dfrac|1|ln<around*|(|b|)>>**<dfrac|t<rsub|i>|a<rsup|L><rsub|i>*>*a<rsup|L><rsub|i>*<around*|(|\<delta\><rsub|ij>-a<rsup|L><rsub|j>|)>|)>=-<dfrac|1|ln<around*|(|b|)>><big|sum><rsub|i=0><rsup|n><around*|(|**<dfrac|t<rsub|i>|a<rsup|L><rsub|i>*>*a<rsup|L><rsub|i>*<around*|(|\<delta\><rsub|ij>-a<rsup|L><rsub|j>|)>|)>=>|<cell|>>|<row|<cell|>|<cell|<dfrac|\<mathd\>c|\<mathd\>z<rsub|j><rsup|L>>=<dfrac|\<mathd\>c|\<mathd\><with|font-series|bold|a><rsup|L>><rsup|T>*\<cdot\><dfrac|\<mathd\><with|font-series|bold|a><rsup|L>|\<mathd\>z<rsub|j><rsup|L>>=-<dfrac|1|ln<around*|(|b|)>>*<matrix|<tformat|<table|<row|<cell|<dfrac|t<rsub|0>|a<rsup|L><rsub|0>*>>>|<row|<cell|<dfrac|t<rsub|1>|a<rsup|L><rsub|1>*>>>|<row|<cell|\<vdots\>>>|<row|<cell|<dfrac|t<rsub|n>|a<rsup|L><rsub|n>>>>>>><rsup|T>\<cdot\><matrix|<tformat|<table|<row|<cell|a<rsup|L><rsub|0>*<around*|(|\<delta\><rsub|0j>-a<rsup|L><rsub|j>|)>>>|<row|<cell|a<rsup|L><rsub|1>*<around*|(|\<delta\><rsub|1j>-a<rsup|L><rsub|j>|)>>>|<row|<cell|\<vdots\>>>|<row|<cell|a<rsup|L><rsub|n>*<around*|(|\<delta\><rsub|n
    j>-a<rsup|L><rsub|j>|)>>>>>>=-<dfrac|1|ln<around*|(|b|)>><big|sum><rsub|i=0><rsup|n><around*|(|**<dfrac|t<rsub|i>|a<rsup|L><rsub|i>*>*a<rsup|L><rsub|i>*<around*|(|\<delta\><rsub|ij>-a<rsup|L><rsub|j>|)>|)>=>|<cell|>>>>
  </eqnarray*>

  The two forms now converge to the same equation. The Kronecker delta can be
  removed from the equation by taking the single case <math|i=j> out of the
  equation and summing across all other cases (<math|i\<neq\>j>).

  <\eqnarray*>
    <tformat|<table|<row|<cell|>|<cell|-<dfrac|1|ln<around*|(|b|)>><around*|(|t<rsub|j>*<around*|(|1-a<rsup|L><rsub|j>|)>-***<big|sum><rsup|n><rsub|i\<neq\>j>t<rsub|i>*a<rsup|L><rsub|<rsup|>j>|)>=-<dfrac|1|ln<around*|(|b|)>><around*|(|t<rsub|j>*<around*|(|1-a<rsup|L><rsub|j>|)>-***a<rsup|L><rsub|<rsup|>j><big|sum><rsup|n><rsub|i\<neq\>j>t<rsub|i>*|)>>|<cell|>>>>
  </eqnarray*>

  As all the outputs of the softmax function sum to 1
  (<math|<big|sum><rsup|n><rsub|i>t<rsub|i>=1>). We get the following that
  allows us to remove the summation: <math|<big|sum><rsup|n><rsub|i\<neq\>j>t<rsub|i>=<big|sum><rsup|n><rsub|i>t<rsub|i>-t<rsub|j>=1-t<rsub|j>>

  <\eqnarray*>
    <tformat|<table|<row|<cell|>|<cell|=-<dfrac|1|ln<around*|(|b|)>><around*|(|t<rsub|j>*<around*|(|1-a<rsup|L><rsub|j>|)>-***a<rsup|L><rsub|<rsup|>j><around*|(|1-t<rsub|j>|)>|)>=>|<cell|>>|<row|<cell|>|<cell|=-<dfrac|1|ln<around*|(|b|)>><around*|(|t<rsub|j>-a<rsub|j><rsup|L>*t<rsub|j>-***a<rsup|L><rsub|<rsup|>j>+a<rsub|j><rsup|L>*t<rsub|j>|)>=-<dfrac|1|ln<around*|(|b|)>><around*|(|t<rsub|j>-***a<rsup|L><rsub|<rsup|>j>|)>>|<cell|>>|<row|<cell|>|<cell|=<dfrac|\<mathd\>c|\<mathd\>z<rsub|j><rsup|L>>=<dfrac|1|ln<around*|(|b|)>><around*|(|a<rsup|L><rsub|<rsup|>j>-t<rsub|j>|)>>|<cell|>>>>
  </eqnarray*>

  Now to use <tfrac|<math|\<mathd\>>c|\<mathd\>z<rsub|j><rsup|L>> in the
  final calculation

  <\eqnarray*>
    <tformat|<table|<row|<cell|>|<cell|<dfrac|\<mathd\>z<rsub|j><rsub|><rsup|L>|\<mathd\>w<rsub|jk><rsup|L>>=a<rsub|k><rsup|L-1>>|<cell|>>|<row|<cell|>|<cell|<dfrac|\<mathd\>c|\<mathd\>w<rsub|jk><rsup|L>>=<around*|(|<dfrac|\<mathd\>c|\<mathd\><with|font-series|bold|a><rsup|L>><rsup|T>*\<cdot\><dfrac|\<mathd\><with|font-series|bold|a><rsup|L>|\<mathd\>z<rsub|j><rsup|L>>|)>**<dfrac|\<mathd\>z<rsub|j><rsub|><rsup|L>|\<mathd\>w<rsub|jk><rsup|L>>=<dfrac|\<mathd\>c|\<mathd\>z<rsub|j><rsup|L>>*<dfrac|\<mathd\>z<rsub|j><rsub|><rsup|L>|\<mathd\>w<rsub|jk><rsup|L>>=<dfrac|1|ln<around*|(|b|)>><around*|(|a<rsup|L><rsub|<rsup|>j>-t<rsub|j>|)>**a<rsub|k><rsup|L-1>>|<cell|>>>>
  </eqnarray*>

  \;

  <subsubsection|Matrix form>

  <\eqnarray*>
    <tformat|<table|<row|<cell|>|<cell|<dfrac|\<mathd\>c|\<mathd\>W<rsup|L>>=<around*|(|<dfrac|\<mathd\>c|\<mathd\><with|font-series|bold|a><rsup|L>><rsup|T>\<cdot\><dfrac|\<mathd\><with|font-series|bold|a><rsup|L>|\<mathd\><with|font-series|bold|z><rsup|L>>|)>\<cdot\><dfrac|\<mathd\><with|font-series|bold|z><rsup|L>|\<mathd\>W<rsup|L>><rsup|T>>|<cell|>>|<row|<cell|>|<cell|<dfrac|\<mathd\>c|\<mathd\><with|font-series|bold|a><rsup|L>>=-<dfrac|1|ln<around*|(|b|)>>*<matrix|<tformat|<table|<row|<cell|<dfrac|t<rsub|0>|a<rsup|L><rsub|0>*>>>|<row|<cell|<dfrac|t<rsub|1>|a<rsup|L><rsub|1>*>>>|<row|<cell|\<vdots\>>>|<row|<cell|<dfrac|t<rsub|n>|a<rsup|L><rsub|n>>>>>>>>|<cell|>>|<row|<cell|>|<cell|<dfrac|\<mathd\><with|font-series|bold|a><rsup|L>|\<mathd\><with|font-series|bold|z><rsup|L>>=<matrix|<tformat|<table|<row|<cell|a<rsup|L><rsub|0>*<around*|(|1-a<rsup|L><rsub|0>|)>>|<cell|-a<rsup|L><rsub|0>*a<rsup|L><rsub|1>>|<cell|\<cdots\>>|<cell|-a<rsup|L><rsub|0>*a<rsup|L><rsub|n>>>|<row|<cell|-a<rsup|L><rsub|1>*a<rsup|L><rsub|0>>|<cell|a<rsup|L><rsub|1
    >*<around*|(|1-a<rsup|L><rsub|1>|)>>|<cell|\<cdots\>>|<cell|-a<rsup|L><rsub|1>*a<rsup|L><rsub|n>>>|<row|<cell|\<vdots\>>|<cell|\<vdots\>>|<cell|\<ddots\>>|<cell|\<vdots\>>>|<row|<cell|-a<rsup|L><rsub|n>*a<rsup|L><rsub|0>>|<cell|-a<rsup|L><rsub|n>*a<rsup|L><rsub|1>>|<cell|\<cdots\>>|<cell|a<rsup|L><rsub|n
    >*<around*|(|1-a<rsup|L><rsub|n>|)>>>>>>>|<cell|>>|<row|<cell|>|<cell|<dfrac|\<mathd\>c|\<mathd\><with|font-series|bold|z><rsup|L>>=<dfrac|\<mathd\>c|\<mathd\><with|font-series|bold|a><rsub|><rsup|L>><rsup|T>\<cdot\><dfrac|\<mathd\><with|font-series|bold|a><rsub|><rsup|L>|\<mathd\><with|font-series|bold|z><rsup|L>>=-<dfrac|1|ln<around*|(|b|)>><matrix|<tformat|<table|<row|<cell|t<rsub|0>*<around*|(|1-a<rsup|L><rsub|0>|)>-*t<rsub|1>*a<rsup|L><rsub|<rsup|>0>+\<cdots\>+-*t<rsub|n>*a<rsup|L><rsub|0>>>|<row|<cell|-t<rsub|0>**a<rsup|L><rsub|<rsup|>1>+t<rsub|1>**<around*|(|1-a<rsup|L><rsub|1>|)>+\<cdots\>+-*t<rsub|n>*a<rsup|L><rsub|1>>>|<row|<cell|\<vdots\>>>|<row|<cell|-*t<rsub|0>*a<rsup|L><rsub|n>-t<rsub|1>**a<rsup|><rsup|L><rsub|n>+\<ldots\>+t<rsub|n>**<around*|(|1-a<rsup|L><rsub|n>|)>>>>>>>|<cell|>>>>
  </eqnarray*>

  Please see element form above for the simplification of the summation

  <\eqnarray*>
    <tformat|<table|<row|<cell|>|<cell|=-<dfrac|1|ln<around*|(|b|)>><matrix|<tformat|<table|<row|<cell|t<rsub|0>-***a<rsup|L><rsub|<rsup|>0>>>|<row|<cell|t<rsub|1>-***a<rsup|L><rsub|<rsup|>1>>>|<row|<cell|\<vdots\>>>|<row|<cell|t<rsub|n>-***a<rsup|L><rsub|<rsup|>n>>>>>>=-<dfrac|1|ln<around*|(|b|)>><around*|(|<with|font-series|bold|t>-<with|font-series|bold|a><rsup|L>|)>=<dfrac|1|ln<around*|(|b|)>><around*|(|<with|font-series|bold|a><rsup|L>-<with|font-series|bold|t>|)>>|<cell|>>|<row|<cell|>|<cell|<dfrac|\<mathd\><with|font-series|bold|z><rsup|L>|\<mathd\>W<rsup|L>>=<matrix|<tformat|<table|<row|<cell|<with|font-series||a><rsub|0><rsup|L-1>>>|<row|<cell|<with|font-series||a><rsub|1><rsup|L-1>>>|<row|<cell|\<vdots\>>>|<row|<cell|<with|font-series||a><rsub|n><rsup|L-1>>>>>>>|<cell|>>|<row|<cell|>|<cell|<dfrac|\<mathd\>c|\<mathd\>W<rsup|L>>=<dfrac|\<mathd\>c|\<mathd\><with|font-series|bold|z><rsup|L>>\<cdot\><dfrac|\<mathd\><with|font-series|bold|z><rsup|L>|\<mathd\>W<rsup|L>><rsup|T>=<dfrac|1|ln<around*|(|b|)>>*<matrix|<tformat|<table|<row|<cell|<around*|(|a<rsup|L><rsub|<rsup|>0>-t<rsub|0>***|)>**a<rsub|0><rsup|L-1>>|<cell|<around*|(|a<rsup|L><rsub|<rsup|>0>-t<rsub|0>|)>**a<rsub|1><rsup|L-1>>|<cell|\<cdots\>>|<cell|<around*|(|a<rsup|L><rsub|<rsup|>0>-t<rsub|0>|)>**a<rsub|n><rsup|L-1>>>|<row|<around*|(|a<rsup|L><rsub|1>-t<rsub|1>|)>**a<rsub|0><rsup|L-1>|<cell|<around*|(|a<rsup|L><rsub|1>-t<rsub|1>|)>**a<rsub|1><rsup|L-1>>|<cell|\<cdots\>>|<cell|<around*|(|a<rsup|L><rsub|1>-t<rsub|1>|)>**a<rsub|n><rsup|L-1>>>|<row|<cell|\<vdots\>>|<cell|\<vdots\>>|<cell|\<ddots\>>|<cell|\<vdots\>>>|<row|<cell|<around*|(|a<rsup|L><rsub|n>-t<rsub|n>|)>**a<rsub|0><rsup|L-1>>|<cell|<around*|(|a<rsup|L><rsub|n>-t<rsub|n>|)>**a<rsub|1><rsup|L-1>>|<cell|\<cdots\>>|<cell|<around*|(|a<rsup|L><rsub|n>-t<rsub|n>|)>**a<rsub|n><rsup|L-1>>>>>>>|<cell|>>|<row|<cell|>|<cell|=<dfrac|1|ln<around*|(|b|)>><around*|(|<around*|(|<with|font-series|bold|<with|font-series|bold|a><rsup|L>-t>|)>\<cdot\><with|font-series|bold|a><rsup|L-1<rsup|T>>|)>>|<cell|>>>>
  </eqnarray*>

  <appendix|Cross entropy derivation>

  To get the derivative of cross entropy, the following rules are required:

  <\padded-center>
    Derivative of a scalar function with respect to a vector is a vector
  </padded-center>

  <\eqnarray*>
    <tformat|<table|<row|<cell|>|<cell|<dfrac|\<mathd\>y|\<mathd\><with|font-series|bold|x>>=<matrix|<tformat|<table|<row|<cell|<tfrac|\<mathd\>y|\<mathd\>x<rsub|0>>>>|<row|<cell|<tfrac|\<mathd\>y|\<mathd\>x<rsub|1>>>>|<row|<cell|\<vdots\>>>|<row|<cell|<tfrac|\<mathd\>y|\<mathd\>x<rsub|n>>>>>>>>|<cell|>>>>
  </eqnarray*>

  \;

  <\padded-center>
    Derivative of a logarithm
  </padded-center>

  <\eqnarray*>
    <tformat|<table|<row|<cell|f<around*|(|x|)>=log<rsub|b><around*|(|x|)>>|<with|font-series||>|<cell|f<rprime|'><around*|(|x|)>=<dfrac|1|x*ln<around*|(|b|)>>>>>>
  </eqnarray*>

  \;

  <\padded-center>
    Chain rule
  </padded-center>

  <\eqnarray*>
    <tformat|<table|<row|<cell|c<around*|(|z|)>=f<around*|(|z|)>+g<around*|(|z|)>>|<cell|>|<cell|c<rprime|'><around*|(|z|)>=f<rprime|'><around*|(|z|)>+g<rprime|'><around*|(|z|)>>>>>
  </eqnarray*>

  \;

  The cross entropy function takes a vector as an input and produces a scalar
  as an output so it is a scalar function.

  <\equation*>
    f:\<bbb-R\><rsup|N>\<rightarrow\>\<bbb-R\>
  </equation*>

  The cross entropy function is:

  <\eqnarray*>
    <tformat|<table|<row|<cell|>|<cell|<math-it|<with|font-series||c>>=-<big|sum><rsub|i=0><rsup|n><with|font-series||t<rsub|i><around*|(|log<rsub|b><with|font-series|bold|<around*|(|<with|font-series||a<rsup|><rsub|i>>|)>>|)>>*=-<around*|(|t<rsub|0>*log<rsub|b><with|font-series|bold|<around*|(|<with|font-series||a<rsup|L><rsub|0>>|)>>+t<rsub|1>*<around*|\<nobracket\>|*log<rsub|b><with|font-series|bold|<around*|(|<with|font-series||a<rsup|L><rsub|1>>|)>>|)>+\<cdots\>+t<rsub|n>*log<rsub|b><with|font-series|bold|<around*|(|<with|font-series||a<rsup|L><rsub|n>>|)>>|)>>|<cell|>>>>
  </eqnarray*>

  The derivative of the a scalar function with with respect to a vector will
  be a vector of all the partial derivatives.

  <\eqnarray*>
    <tformat|<table|<row|<cell|>|<cell|<dfrac|\<mathd\><with|font-series||c>|\<mathd\><with|font-series|bold|a><rsup|L>>=<matrix|<tformat|<table|<row|<cell|<dfrac|\<mathd\>c|\<mathd\>a<rsub|0><rsup|L>>>>|<row|<cell|<dfrac|\<mathd\>c|\<mathd\>a<rsub|1><rsup|L>>>>|<row|<cell|\<vdots\>>>|<row|<cell|<dfrac|\<mathd\>c|\<mathd\>a<rsub|n><rsup|L>>>>>>>>|<cell|>>>>
  </eqnarray*>

  So to find the derivative for an element in
  <math|<tfrac|\<mathd\>c|\<mathd\>a<rsub|><rsup|L>>> where <math|j> is the
  element

  <\eqnarray*>
    <tformat|<table|<row|<cell|>|<cell|<dfrac|\<mathd\>c|\<mathd\>a<rsub|j><rsup|L>>=-<dfrac|\<mathd\>|\<mathd\>a<rsub|j><rsup|L>>*<big|sum><rsub|i=0><rsup|n><with|font-series||t<rsub|i>*<around*|\<nobracket\>|log<rsub|b><with|font-series|bold|<around*|(|<with|font-series||a<rsup|L><rsub|i>>|)>>|)>>=-<dfrac|\<mathd\>|\<mathd\>a<rsub|j><rsup|L>>*<around*|(|t<rsub|0>*<around*|\<nobracket\>|log<rsub|b><with|font-series|bold|<around*|(|<with|font-series||a<rsup|L><rsub|0>>|)>>|)>+t<rsub|1>*<around*|\<nobracket\>|log<rsub|b><with|font-series|bold|<around*|(|<with|font-series||a<rsup|L><rsub|1>>|)>>|)>+\<cdots\>+t<rsub|n>*<around*|\<nobracket\>|log<rsub|b><with|font-series|bold|<around*|(|<with|font-series||a<rsup|L><rsub|n>>|)>>|)>|)>>|<cell|>>>>
  </eqnarray*>

  Every term where <math|i\<neq\>j >will be 0 so we can ignore all terms
  except where <math|i=j>:

  <\eqnarray*>
    <tformat|<table|<row|<cell|>|<cell|<dfrac|\<mathd\>c|\<mathd\>a<rsub|j><rsup|L>>=-<dfrac|\<mathd\>|\<mathd\>a<rsub|j><rsup|L>>**<with|font-series||t<rsub|j><around*|(|log<rsub|b><with|font-series|bold|<around*|(|<with|font-series||a<rsup|><rsub|j>>|)>>|)>>=-t<rsub|j>*<dfrac|\<mathd\>|\<mathd\>a<rsub|j><rsup|L>>**<with|font-series||<around*|(|log<rsub|b><with|font-series|bold|<around*|(|<with|font-series||a<rsup|L><rsub|j>>|)>>|)>>=-<dfrac|t<rsub|j>|a<rsup|L><rsub|j>*ln<around*|(|b|)>>=-<dfrac|1|ln<around*|(|b|)>>*<dfrac|t<rsub|j>|a<rsup|L><rsub|j>*>*>|<cell|>>>>
  </eqnarray*>

  So the derivative is:

  <\eqnarray*>
    <tformat|<table|<row|<cell|>|<cell|<dfrac|\<mathd\><with|font-series||c>|\<mathd\><with|font-series|bold|a><rsup|L>>=<matrix|<tformat|<table|<row|<cell|<dfrac|\<mathd\>c|\<mathd\>a<rsub|0><rsup|L>>>>|<row|<cell|<dfrac|\<mathd\>c|\<mathd\>a<rsub|1><rsup|L>>>>|<row|<cell|\<vdots\>>>|<row|<cell|<dfrac|\<mathd\>c|\<mathd\>a<rsub|n><rsup|L>>>>>>>=-<dfrac|1|ln<around*|(|b|)>>*<matrix|<tformat|<table|<row|<cell|<dfrac|t<rsub|0>|a<rsup|L><rsub|0>*>>>|<row|<cell|<dfrac|t<rsub|1>|a<rsup|L><rsub|1>*>>>|<row|<cell|\<vdots\>>>|<row|<cell|<dfrac|t<rsub|n>|a<rsup|L><rsub|n>>>>>>>=-<dfrac|1|ln<around*|(|b|)>>*<dfrac|<with|font-series|bold|t>|<with|font-series|bold|a><rsup|L>>>|<cell|>>>>
  </eqnarray*>

  <new-page*><appendix|Softmax derivation>

  To get the derivative of softmax, the following rules are required:

  <\padded-center>
    Derivative of a vector function with respect to a vector is a jacobian
    matrix
  </padded-center>

  <\eqnarray*>
    <tformat|<table|<row|<cell|>|<cell|<dfrac|\<mathd\><with|font-series|bold|y>|\<mathd\><with|font-series|bold|x>>=<matrix|<tformat|<table|<row|<cell|<dfrac|\<mathd\><with|font-series||<with|font-series||y<rsub|0>>>|\<mathd\><with|font-series||x<rsub|0>>>>|<cell|<dfrac|\<mathd\><with|font-series||<with|font-series||y<rsub|1>>>|\<mathd\><with|font-series||x<rsub|0>>>>|<cell|\<cdots\>>|<cell|<dfrac|\<mathd\><with|font-series||<with|font-series||y<rsub|n>>>|\<mathd\><with|font-series||x<rsub|0>>>>>|<row|<cell|<dfrac|\<mathd\><with|font-series||<with|font-series||y<rsub|0>>>|\<mathd\><with|font-series||x<rsub|1>>>>|<cell|<dfrac|\<mathd\><with|font-series||<with|font-series||y<rsub|1>>>|\<mathd\><with|font-series||x<rsub|1>>>>|<cell|\<cdots\>>|<cell|<dfrac|\<mathd\><with|font-series||<with|font-series||y<rsub|n>>>|\<mathd\><with|font-series||x<rsub|1>>>>>|<row|\<vdots\>|<cell|\<vdots\>>|<cell|\<ddots\>>|<cell|\<vdots\>>>|<row|<cell|<dfrac|\<mathd\><with|font-series||<with|font-series||y<rsub|0>>>|\<mathd\><with|font-series||x<rsub|n>>>>|<cell|<dfrac|\<mathd\><with|font-series||<with|font-series||y<rsub|1>>>|\<mathd\><with|font-series||x<rsub|n>>>>|<cell|\<cdots\>>|<cell|<dfrac|\<mathd\><with|font-series||<with|font-series||y<rsub|n>>>|\<mathd\><with|font-series||x<rsub|n>>>>>>>>>|<cell|>>>>
  </eqnarray*>

  \;

  <\padded-center>
    total derivative is the sum of all partial derivatives
  </padded-center>

  <\eqnarray*>
    <tformat|<table|<row|<cell|>|<cell|f<rprime|'><around*|(|x,y|)>=f<rprime|'><around*|(|x|)>+f<rprime|'><around*|(|y|)>>|<cell|>>>>
  </eqnarray*>

  \;

  <\padded-center>
    law of independence
  </padded-center>

  <\eqnarray*>
    <tformat|<table|<row|<cell|>|<cell|<dfrac|\<mathd\>e<rsup|z>|\<mathd\>e<rsup|x>>=0>|<cell|>>>>
  </eqnarray*>

  \;

  <\padded-center>
    derivative of exponentials
  </padded-center>

  <\eqnarray*>
    <tformat|<table|<row|f<around*|(|x|)>=e<rsup|x>|<cell|>|<cell|f<rprime|'><around*|(|x|)>=e<rsup|x>>>>>
  </eqnarray*>

  \;

  <\padded-center>
    chain rule
  </padded-center>

  <\eqnarray*>
    <tformat|<table|<row|<cell|c<around*|(|z|)>=f<around*|(|z|)>+g<around*|(|z|)>>|<cell|>|<cell|c<rprime|'><around*|(|z|)>=f<rprime|'><around*|(|z|)>+g<rprime|'><around*|(|z|)>>>>>
  </eqnarray*>

  \;

  <\padded-center>
    quotient rule
  </padded-center>

  <\eqnarray*>
    <tformat|<table|<row|<cell|q<around*|(|z|)>=<dfrac|f<around*|(|z|)>|g<around*|(|z|)>>>|<cell|>|<cell|q<rprime|'><around*|(|z|)>=<dfrac|g<around*|(|z|)>*f<rprime|'><around*|(|z|)>-f<around*|(|z|)>*g<rprime|'><around*|(|z|)>|g<around*|(|z|)><rsup|2>>>>>>
  </eqnarray*>

  \;

  The softmax function takes a vector as an input and produces a vector as an
  output so it is a vector function.

  <\equation*>
    f:\<bbb-R\><rsup|N>\<rightarrow\>\<bbb-R\><rsup|N>
  </equation*>

  The softmax function is:

  <\eqnarray*>
    <tformat|<table|<row|<cell|>|<cell|a<rsup|L><rsub|>=<dfrac|e<rsup|<with|font-series||<with|font-series|bold|z><rsup|L><rsub|>><rsup|>>|<big|sum><rsup|n><rsub|i=0><rsub|>e<rsup|<with|font-series||<with|font-series||z><rsup|L><rsub|i>><rsup|>>>=<matrix|<tformat|<table|<row|<cell|<dfrac|e<rsup|<with|font-series||<with|font-series|bol|z><rsup|L><rsub|0>><rsup|>>|<big|sum><rsup|n><rsub|i=0><rsub|>e<rsup|<with|font-series||<with|font-series||z><rsup|L><rsub|i>><rsup|>>>>>|<row|<cell|<dfrac|e<rsup|<with|font-series||<with|font-series|bol|z><rsup|L><rsub|1>><rsup|>>|<big|sum><rsup|n><rsub|i=0><rsub|>e<rsup|<with|font-series||<with|font-series||z><rsup|L><rsub|i>><rsup|>>>>>|<row|<cell|\<vdots\>>>|<row|<cell|<dfrac|e<rsup|<with|font-series||<with|font-series|bol|z><rsup|L><rsub|n>><rsup|>>|<big|sum><rsup|n><rsub|i=0><rsub|>e<rsup|<with|font-series||<with|font-series||z><rsup|L><rsub|n>><rsup|>>>>>>>>>|<cell|>>>>
  </eqnarray*>

  The derivative of the a vector function with with respect to a vector will
  be the jacobian matrix of all the partial derivatives.

  <\eqnarray*>
    <tformat|<table|<row|<cell|>|<cell|<dfrac|\<mathd\><with|font-series|bold|a><rsub|>|\<mathd\><with|font-series|bold|z><rsub|>>=<matrix|<tformat|<table|<row|<cell|<dfrac|\<mathd\><with|font-series|bold|<with|font-series||a<rsup|><rsub|0>>><rsub|>|\<mathd\>z<rsub|0><rsub|>>+<dfrac|\<mathd\><with|font-series|bold|<with|font-series||a<rsub|0>><rsub|>>|\<mathd\>z<rsub|1><rsub|>>+\<ldots\>+<dfrac|\<mathd\><with|font-series|bold|<with|font-series||a<rsub|>>><rsub|0>|\<mathd\>z<rsub|n><rsub|>>>>|<row|<cell|<dfrac|\<mathd\><with|font-series|bold|<with|font-series||a<rsub|1>>><rsub|>|\<mathd\>z<rsub|0><rsub|>>+<dfrac|\<mathd\><with|font-series|bold|<with|font-series||a<rsub|1>><rsub|>>|\<mathd\>z<rsub|1><rsub|>>+\<ldots\>+<dfrac|\<mathd\><with|font-series|bold|<with|font-series||a<rsub|1>>>|\<mathd\>z<rsub|n><rsub|>>>>|<row|<cell|\<vdots\>>>|<row|<cell|<dfrac|\<mathd\><with|font-series|bold|<with|font-series||a<rsub|n>>><rsub|>|\<mathd\>z<rsub|0><rsub|>>+<dfrac|\<mathd\><with|font-series|bold|<with|font-series||a<rsub|n>><rsub|>>|\<mathd\>z<rsub|1><rsub|>>+\<ldots\>+<dfrac|\<mathd\><with|font-series|bold|<with|font-series||a<rsub|n>>><rsub|>|\<mathd\>z<rsub|n><rsub|>>>>>>>>|<cell|>>>>
  </eqnarray*>

  Let's find the partial derivative <math|<frac|\<mathd\><with|font-series||a<rsub|i>>|\<mathd\>z<rsub|j>>>.
  To do this we will break up the numerator and denominator into separate
  functions to make applying the quotient rule easier. Note that
  <math|<frac|\<mathd\>f<around*|(|z<rsub|i>|)>|\<mathd\>z<rsub|>>> \ has two
  separate cases for when <math|j=k> and <math|j\<neq\>k>.

  <\eqnarray*>
    <tformat|<table|<row|<cell|>|<cell|a<rsub|j>=S<around*|(|z<rsub|j>|)>=<dfrac|e<rsup|<with|font-series||z<rsub|j>><rsup|>>|<big|sum><rsup|n><rsub|i=0><rsub|>e<rsup|<with|font-series||z<rsub|i>><rsup|>>>=<dfrac|f<around*|(|z<rsub|j>|)>|g<around*|(|z<rsub|j>|)>>>|>>>
  </eqnarray*>

  <\eqnarray*>
    <tformat|<table|<row|<cell|>|<cell|>|<cell|f<around*|(|z<rsub|j>|)>=e<rsup|<with|font-series||z<rsub|j>><rsup|>>>>|<row|<cell|where
    j=k>|<cell|>|<cell|<dfrac|\<mathd\>f<around*|(|z<rsub|j>|)>|\<mathd\>z<rsub|k>>=<dfrac|\<mathd\>f<around*|(|z<rsub|j>|)>|\<mathd\>z<rsub|j>>=e<rsup|<with|font-series||z<rsub|j>><rsup|>>>>|<row|<cell|where
    j\<neq\>k>|<cell|>|<cell|<dfrac|\<mathd\>f<around*|(|z<rsub|j>|)>|\<mathd\>z<rsub|k>>=0>>|<row|<cell|g<around*|(|z<rsub|j>|)>=<big|sum><rsup|n><rsub|i=0><rsub|>e<rsup|<with|font-series||z<rsub|i>><rsup|>>=e<rsup|<with|font-series||z<rsub|0>><rsup|>>+e<rsup|<with|font-series|bold|<with|font-series||z<rsub|1>>><rsup|>>+\<cdots\>+e<rsup|<with|font-series||z<rsub|n>><rsup|>>>|<cell|>|<cell|<dfrac|\<mathd\>g<around*|(|z<rsub|j>|)>|\<mathd\>z<rsub|k>>=e<rsup|<with|font-series||z<rsub|k>><rsup|>>>>>>
  </eqnarray*>

  <math|when j=k >

  <\eqnarray*>
    <tformat|<table|<row|<cell|>|<cell|<dfrac|\<mathd\>a<rsub|j>|\<mathd\>z<rsub|k>>=<dfrac|\<mathd\>a<rsub|j>|\<mathd\>z<rsub|j>>=<dfrac|\<mathd\>S<around*|(|z<rsub|j>|)>|\<mathd\>z<rsub|j>>=<dfrac|g<around*|(|z<rsub|j>|)>**e<rsup|<with|font-series||z<rsub|j>><rsup|>>-e<rsup|<with|font-series||z<rsub|j>><rsup|>>*e<rsup|<with|font-series||z<rsub|j>><rsup|>>|g<around*|(|z<rsub|j>|)><rsup|2>>=<dfrac|e<rsup|<with|font-series||z<rsub|j>><rsup|>>*<around*|(|g<around*|(|z<rsub|j>|)>**-*e<rsup|<with|font-series||z<rsub|j>><rsup|>>|)>|g<around*|(|z<rsub|j>|)><rsup|2>>=<dfrac|e<rsup|<with|font-series||z<rsub|j>><rsup|>>*|g<around*|(|z<rsub|j>|)><rsup|>>*<dfrac|g<around*|(|z<rsub|j>|)>**-*e<rsup|<with|font-series||z<rsub|j>><rsup|>>|g<around*|(|z<rsub|j>|)><rsup|>>>|<cell|>>|<row|<cell|>|<cell|=<dfrac|e<rsup|<with|font-series||z<rsub|j>><rsup|>>*|g<around*|(|z<rsub|j>|)><rsup|>>*<around*|(|<dfrac|g<around*|(|z<rsub|j>|)>**|g<around*|(|z<rsub|j>|)><rsup|>>-<dfrac|**e<rsup|<with|font-series||z<rsub|j>><rsup|>>|g<around*|(|z<rsub|j>|)><rsup|>>|)>=S<around*|(|z<rsub|j>|)>*<around*|(|1-S<around*|(|z<rsub|j>|)>|)>=a<rsub|j
    >*<around*|(|1-a<rsub|j>|)>>|<cell|>>>>
  </eqnarray*>

  <math|when j\<neq\>k >

  <\eqnarray*>
    <tformat|<table|<row|<cell|>|<cell|<dfrac|\<mathd\>a<rsub|j>|\<mathd\>z<rsub|k>>=<dfrac|\<mathd\>S<around*|(|z<rsub|j>|)>|\<mathd\>z<rsub|k>>=<dfrac|g<around*|(|z<rsub|j>|)>**0-e<rsup|<with|font-series||z<rsub|j>><rsup|>>*e<rsup|<with|font-series||z<rsub|k>><rsup|>>|g<around*|(|z<rsub|j>|)><rsup|2>>=<dfrac|-e<rsup|<with|font-series||z<rsub|j>><rsup|>>*e<rsup|<with|font-series||z<rsub|k>><rsup|>>|g<around*|(|z<rsub|j>|)><rsup|2>>=<dfrac|-e<rsup|<with|font-series||z<rsub|j>><rsup|>>*|g<around*|(|z<rsub|j>|)><rsup|>>*<dfrac|e<rsup|<with|font-series||z<rsub|k>><rsup|>>|g<around*|(|z<rsub|j>|)><rsup|>>=-S<around*|(|z<rsub|j>|)>*S<around*|(|z<rsub|k>|)>=-a<rsub|j>*a<rsub|k>>|<cell|>>>>
  </eqnarray*>

  We can create the jacobian matrix which contains all the partial
  derivatives for all elements. Note <math|I<rsub|n>> is an identity matrix
  of size <math|n>*<math|n>.

  <\eqnarray*>
    <tformat|<table|<row|<cell|>|<cell|<dfrac|\<mathd\><with|font-series|bold|a><rsup|L><rsub|>|\<mathd\><with|font-series|bold|z><rsub|><rsup|L>>=<matrix|<tformat|<table|<row|<cell|a<rsup|L><rsub|0>*<around*|(|1-a<rsup|L><rsub|0>|)>>|<cell|-a<rsup|L><rsub|0>*a<rsup|L><rsub|1>>|<cell|\<cdots\>>|<cell|-a<rsup|L><rsub|0>*a<rsup|L><rsub|n>>>|<row|<cell|-a<rsup|L><rsub|1>*a<rsup|L><rsub|0>>|<cell|a<rsup|L><rsub|1
    >*<around*|(|1-a<rsup|L><rsub|1>|)>>|<cell|\<cdots\>>|<cell|-a<rsup|L><rsub|1>*a<rsup|L><rsub|n>>>|<row|<cell|\<vdots\>>|<cell|\<vdots\>>|<cell|\<ddots\>>|<cell|\<vdots\>>>|<row|<cell|-a<rsup|L><rsub|n>*a<rsup|L><rsub|0>>|<cell|-a<rsup|L><rsub|n>*a<rsup|L><rsub|1>>|<cell|\<cdots\>>|<cell|a<rsup|L><rsub|n
    >*<around*|(|1-a<rsup|L><rsub|n>|)>>>>>>>|<cell|>>|<row|<cell|>|<cell|=<matrix|<tformat|<table|<row|<cell|a<rsup|L><rsub|0>*>|<cell|a<rsup|L><rsub|1>>|<cell|\<cdots\>>|<cell|a<rsup|L><rsub|n>>>>>>*<matrix|<tformat|<table|<row|<cell|<around*|(|1-a<rsup|L><rsub|0>|)>>|<cell|-*a<rsup|L><rsub|<rsup|>1>>|<cell|\<cdots\>>|<cell|-*a<rsup|L><rsub|n>>>|<row|<cell|-*a<rsup|L><rsub|<rsup|>0>>|<cell|*<around*|(|1-a<rsup|L><rsub|1>|)>>|<cell|\<cdots\>>|<cell|-*a<rsup|L><rsub|n>>>|<row|<cell|\<vdots\>>|<cell|\<vdots\>>|<cell|\<ddots\>>|<cell|\<vdots\>>>|<row|<cell|-*a<rsup|L><rsub|0>>|<cell|-*a<rsup|><rsup|L><rsub|1>>|<cell|\<cdots\>>|<cell|*<around*|(|1-a<rsup|L><rsub|n>|)>>>>>>>|<cell|>>|<row|<cell|>|<cell|=<matrix|<tformat|<table|<row|<cell|a<rsup|L><rsub|0>*>|<cell|a<rsup|L><rsub|1>>|<cell|\<cdots\>>|<cell|a<rsup|L><rsub|n>>>>>>*<around*|(|I<rsub|n>-<matrix|<tformat|<table|<row|<cell|a<rsup|L><rsub|0>>|<cell|*a<rsup|L><rsub|1>>|<cell|\<cdots\>>|<cell|a<rsup|L><rsub|n>>>|<row|<cell|a<rsup|L><rsub|0>>|<cell|a<rsup|L><rsub|1>>|<cell|\<cdots\>>|<cell|a<rsup|L><rsub|n>>>|<row|<cell|\<vdots\>>|<cell|\<vdots\>>|<cell|\<ddots\>>|<cell|\<vdots\>>>|<row|<cell|*a<rsup|L><rsub|0>>|<cell|a<rsup|L><rsub|1>>|<cell|\<cdots\>>|<cell|*a<rsup|L><rsub|n>>>>>>|)>*>|<cell|>>>>
  </eqnarray*>

  This can also be written with the Kronecker delta where
  <math|\<delta\><rsub|jk><choice|<tformat|<table|<row|<cell|1 if
  j=k>>|<row|<cell|0 if j\<neq\>k>>>>>>

  <\eqnarray*>
    <tformat|<table|<row|<cell|>|<cell|<dfrac|\<mathd\><with|font-series||a><rsub|j><rsup|L>|\<mathd\><with|font-series||z<rsub|k>><rsup|L>>=a<rsup|L><rsub|j>*<around*|(|\<delta\><rsub|j*k>-a<rsup|L><rsub|k>|)>>|<cell|>>>>
  </eqnarray*>

  \;

  \;
</body>

<\initial>
  <\collection>
    <associate|page-height|auto>
    <associate|page-medium|paper>
    <associate|page-type|a4>
    <associate|page-width|auto>
  </collection>
</initial>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|2>>
    <associate|auto-10|<tuple|5.1|6>>
    <associate|auto-11|<tuple|5.1.1|7>>
    <associate|auto-12|<tuple|5.1.2|9>>
    <associate|auto-13|<tuple|5.2|11>>
    <associate|auto-14|<tuple|5.2.1|?>>
    <associate|auto-15|<tuple|5.2.2|?>>
    <associate|auto-16|<tuple|A|?>>
    <associate|auto-17|<tuple|B|?>>
    <associate|auto-2|<tuple|2|3>>
    <associate|auto-3|<tuple|2.1|3>>
    <associate|auto-4|<tuple|2.2|3>>
    <associate|auto-5|<tuple|3|4>>
    <associate|auto-6|<tuple|3.1|4>>
    <associate|auto-7|<tuple|3.2|4>>
    <associate|auto-8|<tuple|4|5>>
    <associate|auto-9|<tuple|5|6>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|1<space|2spc>Notation>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|2<space|2spc>Loss/Cost
      Function> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2><vspace|0.5fn>

      <with|par-left|<quote|1tab>|2.1<space|2spc>Squared error
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-3>>

      <with|par-left|<quote|1tab>|2.2<space|2spc>Cross entropy
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-4>>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|3<space|2spc>Activation
      Function> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-5><vspace|0.5fn>

      <with|par-left|<quote|1tab>|3.1<space|2spc>Sigmoid
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-6>>

      <with|par-left|<quote|1tab>|3.2<space|2spc>Softmax
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-7>>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|4<space|2spc>Matrix
      weights> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-8><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|5<space|2spc>Backpropigation>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-9><vspace|0.5fn>

      <with|par-left|<quote|1tab>|5.1<space|2spc>Sigmoid and squared error
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-10>>

      <with|par-left|<quote|1tab>|5.2<space|2spc>Softmax and cross entropy
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-11>>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|Appendix
      A<space|2spc>Cross entropy derivation>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-12><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|Appendix
      B<space|2spc>Softmax derivation> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-13><vspace|0.5fn>
    </associate>
  </collection>
</auxiliary>