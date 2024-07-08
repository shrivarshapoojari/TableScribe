import React, { useState } from 'react';
import axios from 'axios';
import toast, { ToastBar, Toaster } from 'react-hot-toast';

const Card = () => {
    const [selectedFile, setSelectedFile] = useState(null);
    const [imagePreview, setImagePreview] = useState(null);
    const [fileUrl, setFileUrl] = useState(null);

    const handleFileChange = (file) => {
        setSelectedFile(file);

        // Preview image
        const reader = new FileReader();
        reader.onloadend = () => {
            setImagePreview(reader.result);
        };
        if (file) {
            reader.readAsDataURL(file);
        }
    };

    const handleUpload = async () => {
        if (!selectedFile) {
            toast.error("No file selected");
            return;
        }

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            toast.loading("Hang on .. Analyzing Image");
            const response = await axios.post('http://127.0.0.1:5000/upload', formData, {
                responseType: 'blob'
            });

            toast.dismiss();
            toast.success("Table Conversion Success");

            const url = window.URL.createObjectURL(new Blob([response.data]));
            setFileUrl(url);
        } catch (error) {
            toast.dismiss();
            if (error.response && error.response.status === 400) {
                const reader = new FileReader();
                reader.onloadend = () => {
                    const text = reader.result;
                    if (text === 'No tables detected') {
                        toast.error('No tables detected in the image');
                    } else {
                        toast.error('Error uploading file');
                    }
                };
                reader.readAsText(error.response.data);
            } else {
                toast.error('Error uploading file');
            }
            console.error('Error uploading file:', error);
        }
    };

    const handleRemoveImage = () => {
        setSelectedFile(null);
        setImagePreview(null);
        setFileUrl(null);
    };

    const handleDrop = (event) => {
        event.preventDefault();
        const file = event.dataTransfer.files[0];
        handleFileChange(file);
    };

    const handleDragOver = (event) => {
        event.preventDefault();
    };

    return (
        <div>
            <Toaster />
            <div className="min-h-[90vh] flex flex-col items-center justify-center p-5">
                <div 
                    className="max-w-lg w-full bg-white p-12 rounded-xl shadow-2xl shadow-purple-600"
                    onDrop={handleDrop}
                    onDragOver={handleDragOver}
                >
                    <div className="relative mb-6">
                        {imagePreview ? (
                            <div className="mb-4 flex items-center justify-center">
                                <img src={imagePreview} alt="Uploaded" className="rounded-lg max-h-60 mx-auto mb-4" />
                            </div>
                        ) : (
                            <div className="border-dashed border-2 border-gray-400 p-6 rounded-lg text-center">
                                <span className="text-lg text-gray-500">Click Browse Image or Drop Image Here</span>
                            </div>
                        )}
                        <div className="flex items-center justify-center space-x-4">
                            <label htmlFor="file-upload" className="block mt-4">
                                <input id="file-upload" type="file" className="hidden" onChange={(e) => handleFileChange(e.target.files[0])} />
                                <button
                                    className="bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline"
                                    onClick={() => document.getElementById('file-upload').click()}
                                >
                                    {imagePreview ? 'Change Image' : 'Browse Image'}
                                </button>
                            </label>
                            {imagePreview && (
                                <button
                                    className="bg-red-500 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline mt-4"
                                    onClick={handleRemoveImage}
                                >
                                    Remove Image
                                </button>
                            )}
                        </div>
                    </div>
                    {imagePreview && (
                        <div className="text-center">
                            <button
                                className="bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline"
                                onClick={handleUpload}
                            >
                                Upload
                            </button>
                        </div>
                    )}
                    {fileUrl && (
                        <div className="text-center mt-4">
                            <a
                                href={fileUrl}
                                download="output.xlsx"
                                className="bg-green-500 hover:bg-green-600 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline"
                                onClick={() => { setFileUrl(null); }}
                            >
                                Download File
                            </a>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default Card;
