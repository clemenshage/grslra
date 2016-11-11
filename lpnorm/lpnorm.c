#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <immintrin.h>
#include <omp.h>
#include <Python.h>
#include <numpy/arrayobject.h>


static PyObject *lpnorm_func(PyObject* self, PyObject* args)
{
    PyObject *arr_obj = NULL;
    PyObject *arr_arr = NULL;
    PyArrayObject *arr = NULL;
    double p;
    double mu;
    int sse;
    int openmp;

    int result = PyArg_ParseTuple(
        args,
        "Oddii",
        &arr_obj,
        &mu,
        &p,
        &sse,
        &openmp
    );

    if (!result) {
        return NULL;
    }

    arr_arr = PyArray_FROM_OTF(arr_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (arr_arr == NULL) {
        return NULL;
    }
    arr = (PyArrayObject*) arr_arr;

    int ndim = PyArray_NDIM(arr);
    if (ndim > 2) {
        PyErr_SetString(PyExc_ValueError,
            "The input array should be 1- or 2-dimensional");
        return NULL;
    }

    npy_intp *dims = PyArray_DIMS(arr);

    int nelem = 0;

    if (ndim == 1) {
        nelem = dims[0];
    } else {
        nelem = dims[0]*dims[1];
    }
    double *buffer = PyArray_DATA(arr);

    double val = 0;
    double p_half = p / 2;
    if (openmp) {
        if (sse) {
            __m128d mus = _mm_load_pd1(&mu);
            __m128d p_halfs = _mm_load_pd1(&p_half);

            #pragma omp parallel for reduction(+:val)
            for (int i=0; i<(nelem&0xFFFFFFFE); i+=2) {
                double temp[2];
                __m128d in = _mm_load_pd(buffer+i);

                _mm_store_pd(temp, _mm_add_pd(_mm_mul_pd(in, in), mus));

                temp[0] = log(temp[0]);
                temp[1] = log(temp[1]);

                _mm_store_pd(temp, _mm_mul_pd(_mm_load_pd(temp), p_halfs));

                val += exp(temp[0]);
                val += exp(temp[1]);
            }
            if (nelem & 1) {
                double t = buffer[nelem-1];
                val += exp(log(t * t + mu) * p_half);
            }
        } else {
            #pragma omp parallel for reduction(+:val)
            for (int i=0; i<nelem; i++) {
                double t = buffer[i];
                val += exp(log(t * t + mu) * p_half);
            }
        }
    } else {
        if (sse) {
            double temp[2];
            __m128d mus = _mm_load_pd1(&mu);
            __m128d p_halfs = _mm_load_pd1(&p_half);

            while (nelem>1) {
                __m128d in = _mm_load_pd(buffer);

                _mm_store_pd(temp, _mm_add_pd(_mm_mul_pd(in, in), mus));

                temp[0] = log(temp[0]);
                temp[1] = log(temp[1]);

                _mm_store_pd(temp, _mm_mul_pd(_mm_load_pd(temp), p_halfs));

                val += exp(temp[0]);
                val += exp(temp[1]);

                buffer += 2;
                nelem -= 2;
            }
            if (nelem) {
                double t = buffer[nelem-1];
                val += exp(log(t * t + mu) * p_half);
            }
        } else {
            while (nelem--) {
                double t = *(buffer++);
                val += exp(log(t * t + mu) * p_half);
            }
        }
    }

    Py_DECREF(arr_arr);
    return Py_BuildValue("d", val);
}


static PyObject *lpnormgrad_func(PyObject* self, PyObject* args)
{
    PyObject *arr_obj = NULL;
    PyObject *arr_arr = NULL;
    PyArrayObject *arr = NULL;

    PyObject *out_obj = NULL;
    PyObject *out_arr = NULL;
    PyArrayObject *out = NULL;

    double p;
    double mu;
    int openmp;

    int result = PyArg_ParseTuple(
        args,
        "OddiO",
        &arr_obj,
        &mu,
        &p,
        &openmp,
        &out_obj
    );

    if (!result) {
        return NULL;
    }

    arr_arr = PyArray_FROM_OTF(arr_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (arr_arr == NULL) {
        return NULL;
    }
    arr = (PyArrayObject*) arr_arr;

    int ndim = PyArray_NDIM(arr);
    if (ndim > 2) {
        PyErr_SetString(PyExc_ValueError,
            "The input array should be 1- or 2-dimensional");
        return NULL;
    }

    npy_intp *dims = PyArray_DIMS(arr);

    out_arr = PyArray_FROM_OTF(out_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (out_arr == NULL) {
        return NULL;
    }
    out = (PyArrayObject*) out_arr;

    int ndim_out = PyArray_NDIM(out);
    if (ndim_out > 2) {
        PyErr_SetString(PyExc_ValueError,
            "The output array should be 1- or 2-dimensional");
        return NULL;
    }

    if (ndim != ndim_out) {
    PyErr_SetString(PyExc_ValueError,
            "input and output dimensions should match");
        return NULL;
    }

    int nelem = 0;

    if (ndim == 1) {
        nelem = dims[0];
    } else {
        nelem = dims[0]*dims[1];
    }
    double *buffer = PyArray_DATA(arr);
    double *buffer_out = PyArray_DATA(out_arr);

    double p_half = p / 2 - 1;
    if (openmp) {
            #pragma omp parallel for
            for (int i=0; i<nelem; i++) {
                double t = buffer[i];
                buffer_out[i] = p * t * exp(log(t * t + mu) * p_half);
        }
    } else {
            for (int i=0; i<nelem; i++) {
                double t = buffer[i];
                buffer_out[i] = p * t * exp(log(t * t + mu) * p_half);
            }
        }
    Py_DECREF(arr_arr);
    Py_DECREF(out_arr);
    Py_RETURN_NONE;
}

static PyMethodDef lpnorm_methods[] =
{
     {"lpnorm", lpnorm_func, METH_VARARGS, "calculate fast lp norm in pure C"},
     {"lpnormgrad", lpnormgrad_func, METH_VARARGS, "calculate fast lp norm gradient in pure C"},
     {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_lpnorm",
    NULL,
    -1,
    lpnorm_methods
};

PyObject *
PyInit__lpnorm(void)
{
    import_array();
    PyObject *module = PyModule_Create(&moduledef);
    return module;
}
#else
void init_lpnorm(void)
{
    import_array();
    Py_InitModule("_lpnorm", lpnorm_methods);
    return;
}

#endif
