#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <immintrin.h>
#include <omp.h>
#include <Python.h>
#include <numpy/arrayobject.h>

#include <cblas.h>


static PyObject *smmprod_func(PyObject* self, PyObject* args)
{
    PyObject *A_obj = NULL;
    PyObject *A_arr = NULL;
    PyArrayObject *A = NULL;
    
    PyObject *B_obj = NULL;
    PyObject *B_arr = NULL;
    PyArrayObject *B = NULL;
    
    PyObject *Omega_obj = NULL;
    PyObject *Omega_row_arr = NULL;
    PyObject *Omega_col_arr = NULL;
    PyObject *Omega_seq = NULL;
    PyArrayObject *Omega_row = NULL;
    PyArrayObject *Omega_col = NULL;

    PyObject *Out_obj = NULL;
    PyObject *Out_arr = NULL;
    PyArrayObject *Out = NULL;
    
    int result = PyArg_ParseTuple(
        args,
        "OOOO",
        &A_obj,
        &B_obj,
        &Omega_obj,
        &Out_obj
    );

    if (!result) {
        return NULL;
    }

    A_arr = PyArray_FROM_OTF(A_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (A_arr == NULL) {
        return NULL;
    }
    A = (PyArrayObject*) A_arr;

    B_arr = PyArray_FROM_OTF(B_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (B_arr == NULL) {
        return NULL;
    }
    B = (PyArrayObject*) B_arr;

    Omega_seq = PySequence_Fast(Omega_obj, "expected a sequence");

    Omega_row_arr = PyArray_FROM_OTF(PySequence_Fast_GET_ITEM(Omega_seq, 0), NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (Omega_row_arr == NULL) {
        return NULL;
    }
    Omega_row = (PyArrayObject*) Omega_row_arr;

    Omega_col_arr = PyArray_FROM_OTF(PySequence_Fast_GET_ITEM(Omega_seq, 1), NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (Omega_col_arr == NULL) {
        return NULL;
    }
    Omega_col = (PyArrayObject*) Omega_col_arr;

    Out_arr = PyArray_FROM_OTF(Out_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (Out_arr == NULL) {
        return NULL;
    }
    Out = (PyArrayObject*) Out_arr;

    int ndimA = PyArray_NDIM(A);
    if (ndimA != 2) {
        PyErr_SetString(PyExc_ValueError,
            "A should be 2-dimensional");
        return NULL;
    }

    int ndimB = PyArray_NDIM(B);
    if (ndimB != 2) {
        PyErr_SetString(PyExc_ValueError,
            "B should be 2-dimensional");
        return NULL;
    }

//    int ndimOmega = PyArray_NDIM(B);
//    if (ndimOmega != 2) {
//        PyErr_SetString(PyExc_ValueError,
//            "Omega should be 2-dimensional");
//        return NULL;
//    }

    npy_intp *dimsA = PyArray_DIMS(A);
    npy_intp *dimsB = PyArray_DIMS(B);
    npy_intp *dimsOmega_row = PyArray_DIMS(Omega_row);
    npy_intp *dimsOmega_col = PyArray_DIMS(Omega_col);

    if (dimsOmega_row[0] != dimsOmega_col[0]) {
        PyErr_SetString(PyExc_ValueError,
            "index arrays should have equal length");
        return NULL;
    }

    if (dimsA[1] != dimsB[0] ) {
        PyErr_SetString(PyExc_ValueError,
            "Inner dimensions of A and B do not match");
        return NULL;
    }

    double *bufferA = PyArray_DATA(A);
    double *bufferB = PyArray_DATA(B);
    double *bufferOmega_row = PyArray_DATA(Omega_row);
    double *bufferOmega_col = PyArray_DATA(Omega_col);
    double *bufferOut = PyArray_DATA(Out);

    //int m = dimsA[0];
    int k = dimsA[1];
    int n = dimsB[1];
    int j = dimsOmega_row[0];

    // Remove this pragma for single threaded execution
    #pragma omp parallel for
    for (int i=0; i<j; i++) {
        double *iA = bufferA + k * (int)(bufferOmega_row[i]);
        double *iB = bufferB + (int)(bufferOmega_col[i]);

        /*
         * The plain C version, without blas
        double val = 0;
        for (int x=0; x<k; x++) {
            val += *iA * *iB;
            iA += 1;
            iB += n;
        }
        *(bufferOut++) = val;
        */
        bufferOut[i] = cblas_ddot(k, iA, 1, iB, n);
    }

    Py_DECREF(A_arr);
    Py_DECREF(B_arr);
    Py_DECREF(Omega_row_arr);
    Py_DECREF(Omega_col_arr);
    Py_DECREF(Omega_seq);
    Py_DECREF(Out);
    Py_RETURN_NONE;
}

static PyMethodDef smmprod_methods[] =
{
     {"smmprod", smmprod_func, METH_VARARGS, "calculate fast smmprod in pure C"},
     {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_smmprod",
    NULL,
    -1,
    smmprod_methods
};

PyObject *
PyInit__smmprod(void)
{
    import_array();
    PyObject *module = PyModule_Create(&moduledef);
    return module;
}
#else
void init_smmprod(void)
{
    import_array();
    Py_InitModule("_smmprod", smmprod_methods);
    return;
}

#endif
