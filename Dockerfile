# Start from a core stack version
FROM jupyter/scipy-notebook:latest
# Install from requirements.txt file
RUN git clone https://github.com/LoganAMorrison/Hazma.git /tmp/Hazma
RUN pip install yapf
RUN pip install jupyter_contrib_nbextensions
RUN jupyter contrib nbextension install --user
RUN jupyter nbextension enable codefolding/main
RUN jupyter nbextension enable code_prettify/code_prettify
RUN jupyter nbextension enable init_cell/main
RUN jupyter nbextension enable freeze/main
RUN jupyter nbextension enable hide_input/main
RUN jupyter nbextension enable jupyter-js-widgets/extension
RUN jupyter nbextension enable contrib_nbextensions_help_item/main
RUN jupyter nbextension enable collapsible_headings/main
RUN jupyter nbextension enable toc2/main
RUN pip install --requirement /tmp/Hazma/requirements.txt
RUN ls /tmp/
RUN pip install /tmp/Hazma
RUN fix-permissions $CONDA_DIR 
RUN fix-permissions /home/$NB_USER